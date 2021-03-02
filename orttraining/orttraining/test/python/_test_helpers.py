import numpy as np
import os
import sys
import torch

from numpy.testing import assert_allclose
from onnxruntime.training import orttrainer
from onnxruntime.capi.ort_trainer import ORTTrainer as Legacy_ORTTrainer


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
    for inp in model._onnx_training.graph.input:
        shape = inp.type.tensor_type.shape
        if shape:
            for dim in shape.dim:
                if dim.dim_param and not isinstance(dim.dim_param, str):
                    return False

    # Check outputs
    for out in model._onnx_training.graph.output:
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

def assert_torch_allclose(ort_tensor, pt_tensor, use_fp16=False, rtol=None, atol=None):
    if rtol is None:
        rtol = 1e-3 if use_fp16 else 1e-5
    if atol is None:
        atol = 1e-3 if use_fp16 else 1e-6
    
    assert torch.allclose(ort_tensor, pt_tensor, rtol=rtol, atol=atol)

def assert_gradients_match_and_reset_gradient(ort_model, pt_model, use_fp16=False, reset_gradient=True, rtol=None, atol=None):
    if rtol is None:
        rtol = 1e-3 if use_fp16 else 1e-5
    if atol is None:
        atol = 1e-3 if use_fp16 else 1e-6

    ort_named_params = list(ort_model.named_parameters())
    pt_named_params = list(pt_model.named_parameters())
    assert len(ort_named_params) == len(pt_named_params)

    for ort_named_param, pt_named_param in zip(ort_named_params, pt_named_params):
        ort_name, ort_param = ort_named_param
        pt_name, pt_param = pt_named_param

        assert pt_name in ort_name
        assert torch.allclose(ort_param.grad, pt_param.grad, rtol=rtol, atol=atol)

        if reset_gradient:
            ort_param.grad = None
            pt_param.grad = None
