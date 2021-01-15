import math
import os
import sys
import subprocess
import copy
import numpy as np
from numpy.testing import assert_allclose
import torch
import onnx

from onnxruntime.training import optim, _utils

def _single_run(execution_file, scenario, checkopint_dir = None):
    print("Running single_run scenario: ", scenario)

    cmd = [sys.executable, execution_file]
    if scenario:
        cmd += ['--scenario', scenario]
    if checkopint_dir:
        cmd += ['--checkpoint_dir', checkopint_dir]
    assert subprocess.call(cmd) == 0

def _distributed_run(execution_file, scenario, checkopint_dir = None):
    print("Running distributed_run scenario: ", scenario)

    ngpus = torch.cuda.device_count()
    cmd = ['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable, execution_file]
    if scenario:
        cmd += ['--scenario', scenario]
    if checkopint_dir:
        cmd += ['--checkpoint_dir', checkopint_dir]
    assert subprocess.call(cmd) == 0

def is_windows():
    return sys.platform.startswith("win")

def run_subprocess(args, cwd=None, capture=False, dll_path=None,
                   shell=False, env={}, log=None):
    if log:
        log.info("Running subprocess in '{0}'\n{1}".format(
            cwd or os.getcwd(), args))
    my_env = os.environ.copy()
    if dll_path:
        if is_windows():
            my_env["PATH"] = dll_path + os.pathsep + my_env["PATH"]
        else:
            if "LD_LIBRARY_PATH" in my_env:
                my_env["LD_LIBRARY_PATH"] += os.pathsep + dll_path
            else:
                my_env["LD_LIBRARY_PATH"] = dll_path

    stdout, stderr = (subprocess.PIPE, subprocess.STDOUT) if capture else (
        None, None)
    my_env.update(env)
    completed_process = subprocess.run(
        args, cwd=cwd, check=True, stdout=stdout, stderr=stderr,
        env=my_env, shell=shell)
    
    if log:
        log.debug("Subprocess completed. Return code=" +
                str(completed_process.returncode))
    return completed_process


def legacy_constant_lr_scheduler(global_step, initial_lr, total_steps, warmup):
    num_warmup_steps = warmup * total_steps
    if global_step < num_warmup_steps:
        new_lr = initial_lr * float(global_step) / float(max(1, num_warmup_steps))
    else:
        new_lr = initial_lr
    return new_lr


def legacy_cosine_lr_scheduler(global_step, initial_lr, total_steps, warmup, cycles):
    num_warmup_steps = warmup * total_steps
    if global_step < num_warmup_steps:
        new_lr = initial_lr * float(global_step) / float(max(1, num_warmup_steps))
    else:
        progress = float(global_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
        new_lr = initial_lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(cycles) * 2.0 * progress)))
    return new_lr



def legacy_linear_lr_scheduler(global_step, initial_lr, total_steps, warmup):
    num_warmup_steps = warmup * total_steps
    if global_step < num_warmup_steps:
        new_lr = initial_lr * float(global_step) / float(max(1, num_warmup_steps))
    else:
        new_lr = initial_lr * max(0.0, float(total_steps - global_step) / float(max(1, total_steps - num_warmup_steps)))
    return new_lr


def legacy_poly_lr_scheduler(global_step, initial_lr, total_steps, warmup, power, lr_end):
    num_warmup_steps = warmup * total_steps
    if global_step < num_warmup_steps:
        new_lr = initial_lr * float(global_step) / float(max(1, num_warmup_steps))
    elif global_step > total_steps:
        new_lr = lr_end
    else:
        lr_range = initial_lr - lr_end
        decay_steps = total_steps - num_warmup_steps
        pct_remaining = 1 - (global_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining ** power + lr_end
        new_lr = decay
    return new_lr


def generate_dummy_optim_state(model, optimizer):
    np.random.seed(0)
    if not (isinstance(optimizer, optim.AdamConfig) or isinstance(optimizer, optim.LambConfig)):
        return dict()

    moment_keys = ["Moment_1", "Moment_2"]
    uc_key = "Update_Count"
    step_key = "Step"
    shared_state_key = "shared_optimizer_state"

    optim_state = dict()
    weight_shape_map = dict()
    if isinstance(model, torch.nn.Module):
        weight_shape_map = {name: param.size() for name, param in model.named_parameters()}
    elif isinstance(model, onnx.ModelProto):
        weight_shape_map = {n.name: n.dims for n in model.graph.initializer}
    else:
        raise ValueError("'model' must be either 'torch.nn.Module' or 'onnx.ModelProto'")

    for weight_name, weight_shape in weight_shape_map.items():
        per_weight_state = dict()
        for moment in moment_keys:
            per_weight_state[moment] = np.random.uniform(-2, 2, weight_shape).astype(np.float32)
        if isinstance(optimizer, optim.AdamConfig):
            per_weight_state[uc_key] = np.full([1], 5, dtype=np.int64)
        optim_state[weight_name] = copy.deepcopy(per_weight_state)
    if isinstance(optimizer, optim.LambConfig):
        step_val = np.full([1], 5, dtype=np.int64)
        optim_state[shared_state_key] = {step_key: step_val}
    return {
        'optimizer': optim_state,
        'trainer_options': {
            'optimizer_name': optimizer.name
        }
    }

def _load_pytorch_transformer_model(device, dynamic_axes=False, legacy_api=False):
    # Loads external Pytorch TransformerModel into utils
    pytorch_transformer_path = os.path.join('samples', 'python', 'pytorch_transformer')
    pt_model_path = os.path.join(pytorch_transformer_path, 'pt_model.py')
    pt_model = _utils.import_module_from_file(pt_model_path)
    ort_utils_path = os.path.join(pytorch_transformer_path, 'ort_utils.py')
    ort_utils = _utils.import_module_from_file(ort_utils_path)
    utils_path = os.path.join(pytorch_transformer_path, 'utils.py')
    utils = _utils.import_module_from_file(utils_path)

    # Modeling
    model = pt_model.TransformerModel(28785, 200, 2, 200, 2, 0.2).to(device)
    my_loss = ort_utils.my_loss
    if legacy_api:
        if dynamic_axes:
            model_desc = ort_utils.legacy_transformer_model_description_dynamic_axes()
        else:
            model_desc = ort_utils.legacy_transformer_model_description()
    else:
        if dynamic_axes:
            model_desc = ort_utils.transformer_model_description_dynamic_axes()
        else:
            model_desc = ort_utils.transformer_model_description()


    # Preparing data
    train_data, val_data, test_data = utils.prepare_data(device, 20, 20)
    return model, model_desc, my_loss, utils.get_batch, train_data, val_data, test_data

def assert_all_states_close_ort(state_dict_pre_checkpoint, state_dict_post_checkpoint, reshape_states=False):
    """Assert that the two ORTTrainer (hierarchical) state dictionaries are very close for all states"""

    assert ('model' in state_dict_pre_checkpoint) == ('model' in state_dict_post_checkpoint)
    assert ('optimizer' in state_dict_pre_checkpoint) == ('optimizer' in state_dict_post_checkpoint)

    if 'model' in state_dict_pre_checkpoint:
        for model_state_key in state_dict_pre_checkpoint['model']['full_precision']:
            if reshape_states:
                assert_allclose(state_dict_pre_checkpoint['model']['full_precision'][model_state_key],
                    state_dict_post_checkpoint['model']['full_precision'][model_state_key]\
                        .reshape(state_dict_pre_checkpoint['model']['full_precision'][model_state_key].shape))
            else:
                assert_allclose(state_dict_pre_checkpoint['model']['full_precision'][model_state_key],
                    state_dict_post_checkpoint['model']['full_precision'][model_state_key])

    if 'optimizer' in state_dict_pre_checkpoint:
        for model_state_key in state_dict_pre_checkpoint['optimizer']:
            for optimizer_state_key in state_dict_pre_checkpoint['optimizer'][model_state_key]:
                if reshape_states:
                    assert_allclose(state_dict_pre_checkpoint['optimizer'][model_state_key][optimizer_state_key],
                        state_dict_post_checkpoint['optimizer'][model_state_key][optimizer_state_key]\
                            .reshape(state_dict_pre_checkpoint['optimizer'][model_state_key][optimizer_state_key].shape))
                else:
                    assert_allclose(state_dict_pre_checkpoint['optimizer'][model_state_key][optimizer_state_key],
                        state_dict_post_checkpoint['optimizer'][model_state_key][optimizer_state_key])

def assert_all_states_close_pytorch(state_dict_pre_checkpoint, pytorch_model):
    """Assert that the state_dict_pre_checkpoint state dictionary is very close to the one extracted from the pytorch model after loading"""

    pytorch_model.load_state_dict(state_dict_pre_checkpoint)
    state_dict_pytorch = pytorch_model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert_allclose(value, state_dict_pre_checkpoint[key])
