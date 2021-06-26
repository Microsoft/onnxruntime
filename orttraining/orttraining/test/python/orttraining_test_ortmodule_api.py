# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# orttraining_test_ortmodule_api.py

import math
import random
import copy
import torch
from transformers import AutoConfig, BertForSequenceClassification, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
import pytest
from time import sleep
import warnings
from unittest.mock import patch
from collections import OrderedDict
from collections import namedtuple
from inspect import signature
import tempfile

from onnxruntime.training.ortmodule import ORTModule, _utils, _io
import _test_helpers

# Import autocasting libs
from torch.cuda import amp

# PyTorch model definitions for tests

class NeuralNetSinglePositionalArgument(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetSinglePositionalArgument, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class NeuralNetMultiplePositionalArgumentsMultiOutputsWithoutDependency(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetMultiplePositionalArgumentsMultiOutputsWithoutDependency, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(input_size, hidden_size)
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.softmax2 = torch.nn.Softmax(dim=1)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def forward(self, input1, input2):
        model_input = input1 + input2
        out1 = self.fc1(model_input)
        out2 = self.fc2(model_input)
        out1 = self.softmax1(out1)
        out2 = self.softmax2(out2)
        out1 = self.relu1(out1)
        out2 = self.relu2(out2)
        return out1, out2

class NeuralNetMultiplePositionalArgumentsMultiOutputsWithDependency(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetMultiplePositionalArgumentsMultiOutputsWithDependency, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.softmax = torch.nn.Softmax(dim=1)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1, input2):
        model_input = input1 + input2
        out1 = self.fc1(model_input)
        out1 = self.softmax(out1)
        # TODO: Using relu here will cause the forward prediction error
        # ORT's Relu output is sharing the same buffer as input,
        # and this buffer is returned as ORTModule's output to Pytorch
        out2 = self.fc2(out1)
        return out1, out2

class NeuralNetMultiplePositionalArguments(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetMultiplePositionalArguments, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1, input2):
        model_input = input1 + input2
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class NeuralNetMultiplePositionalArgumentsVarKeyword(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetMultiplePositionalArgumentsVarKeyword, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1, input2, **kwargs):
        model_input = input1 + input2
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class NeuralNetPositionalArguments(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetPositionalArguments, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, *model_inputs):
        model_input = torch.sum(torch.stack(model_inputs), dim=0)
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class NeuralNetKeywordArguments(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetKeywordArguments, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x=None, y=None, z=None):
        model_input = torch.sum(torch.stack([x, y, z]), dim=0)
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class NeuralNetPositionalAndKeywordArguments(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetPositionalAndKeywordArguments, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, model_input, x=None, y=None, z=None):
        model_input = model_input + torch.sum(torch.stack([x, y, z]), dim=0)
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class NeuralNetSimplePositionalAndKeywordArguments(torch.nn.Module):
    def __init__(self):
        super(NeuralNetSimplePositionalAndKeywordArguments, self).__init__()
        self.a = torch.nn.Parameter(torch.FloatTensor([-1., 1.]))
    def forward(self, x, y=None, z=None):
        if z is not None:
            return torch.mean(self.a) + x + 4 * z
        if y is not None:
            return torch.mean(self.a) + 3 * y
        return torch.mean(self.a) + x

class NeuralNetNonDifferentiableOutput(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetNonDifferentiableOutput, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out1 = self.relu(out)
        out2 = self.fc2(out1)
        mask1 = torch.gt(out1, 0.01)
        mask1 = mask1.long()    # TODO: Casting from bool to float or int will cause the UT failure
                                # True is casted to 1065353216 for Cast(from=bool, to=int), whereas pytorch would give 1
                                # True is casted to -1 for Cast(from=bool, to=float), where as pytorch would give 1.0f
        mask2 = torch.lt(out2, 0.02)
        mask2 = mask2.long()
        
        return out1, mask1, out2, mask2     # intentionally place the non-differentiable output in the middle

class NeuralNetPartialNoGradModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetPartialNoGradModel, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size).requires_grad_(False)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, model_input):
        out = self.relu(self.fc1(model_input))
        out = self.fc2(out)
        return out

class UnusedEndParameterNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(UnusedEndParameterNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.relu = torch.nn.ReLU()
        # fc2 is an unused initializer (which is in the end of initializer list)
        # which will be dropped after export
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.register_buffer("buffer", torch.ones(hidden_size1))

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = out + self.buffer
        return out

class UnusedBeginParameterNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(UnusedBeginParameterNet, self).__init__()

        # fc1 is an unused initializer (which is in the begining of initializer list)
        # which will be dropped after export
        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(input_size, hidden_size2)
        self.register_buffer("buffer", torch.ones(hidden_size2))

    def forward(self, input1):
        out = self.fc2(input1)
        out = self.relu(out)
        out = out + self.buffer
        return out

class UnusedMiddleParameterNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(UnusedMiddleParameterNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.relu = torch.nn.ReLU()
        # fc2 is an unused initializer (which is in the middle of initializer list)
        # which will be dropped after export
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = torch.nn.Linear(hidden_size1, num_classes)
        self.register_buffer("buffer", torch.ones(num_classes))

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc3(out)
        out = out + self.buffer
        return out

class StatelessModel(torch.nn.Module):
    def __init__(self):
        super(StatelessModel, self).__init__()

    def forward(self, x):
        return x

# TODO: This is a workaround for the problem that pytest is still cleaning up the previous test
# while the next task already start. 
@pytest.fixture(autouse=True)
def run_before_tests():
    # wait for 50ms before starting the next test
    sleep(0.05)

def _get_bert_for_sequence_classification_model(device, output_attentions = False, \
    output_hidden_states = False, return_dict = True, hidden_dropout_prob = 0.1, attention_probs_dropout_prob = 0.1):
    """Returns the BertForSequenceClassification pretrained model"""

    config = AutoConfig.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            num_hidden_layers=1,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            hidden_dropout_prob = hidden_dropout_prob, 
            attention_probs_dropout_prob = attention_probs_dropout_prob,
    )
    config.return_dict = return_dict

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        config=config,
    ).to(device)

    return model

def _get_bert_for_sequence_classification_sample_data(device):
    """Returns sample data to be used with BertForSequenceClassification model"""

    input_ids = torch.randint(0, 100, (32, 64), dtype=torch.long, device=device)
    input_mask = torch.randint(0, 100, (32, 64), dtype=torch.long, device=device)
    labels = torch.randint(0, 1, (32,), dtype=torch.long, device=device)

    return input_ids, input_mask, labels

def _get_bert_for_sequence_classification_sample_data_with_random_shapes(device):
    """Returns sample data with random shape to be used with BertForSequenceClassification model"""

    x = random.randint(1,100)
    y = random.randint(1,100)
    input_ids = torch.randint(0, 100, (x, y), dtype=torch.long, device=device)
    input_mask = torch.randint(0, 100, (x, y), dtype=torch.long, device=device)
    labels = torch.randint(0, 1, (x,), dtype=torch.long, device=device)

    return input_ids, input_mask, labels

# ORTModule-API tests

def test_forward_call_single_positional_argument():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(model)
    # Check that the original forward signature is preserved.
    assert signature(model.forward) == signature(ort_model.forward)
    x = torch.randn(N, D_in, device=device)
    # Make sure model runs without any exception
    prediction = ort_model(x)
    assert prediction is not None
    prediction = prediction.sum()
    prediction.backward()

def test_forward_call_multiple_positional_arguments():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetMultiplePositionalArguments(input_size=D_in, hidden_size=H, num_classes=D_out).to(device)
    ort_model = ORTModule(model)
    # Check that the original forward signature is preserved.
    assert signature(model.forward) == signature(ort_model.forward)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)

    # Make sure model runs without any exception
    prediction = ort_model(x, y)
    assert prediction is not None
    prediction = prediction.sum()
    prediction.backward()

def test_forward_call_positional_arguments():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetPositionalArguments(input_size=D_in, hidden_size=H, num_classes=D_out).to(device)
    model = ORTModule(model)
    args = [torch.randn(N, D_in, device=device), torch.randn(N, D_in, device=device), torch.randn(N, D_in, device=device)]

    # Make sure model runs without any exception
    prediction = model(*args)
    assert prediction is not None
    prediction = prediction.sum()
    prediction.backward()

def test_forward_call_keyword_arguments():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetKeywordArguments(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    # Make sure model runs without any exception
    prediction = model(x, y, z)
    assert prediction is not None
    prediction = prediction.sum()
    prediction.backward()

def test_forward_call_positional_and_keyword_arguments():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetPositionalAndKeywordArguments(D_in, H, D_out).to(device)
    model = ORTModule(model)
    a = torch.randn(N, D_in, device=device)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    # Make sure model runs without any exception
    prediction = model(a, x, y, z)
    assert prediction is not None
    prediction = prediction.sum()
    prediction.backward()

@pytest.mark.parametrize("forward_statement", [
    "model(one)",
    "model(x=one)",
    "model(one, None, None)",
    "model(one, None, z=None)",
    "model(one, None)",
    "model(x=one, y=one)",
    "model(y=one, x=one)",
    "model(y=one, z=None, x=one)",
    "model(one, None, z=one)",
    "model(x=one, z=one)",
    "model(one, z=one)",
    "model(one, z=one, y=one)",
    "model(one, one, one)",
    "model(one, None, one)",
    "model(z=one, x=one, y=one)",
    "model(z=one, x=one, y=None)"
])
def test_compare_pytorch_forward_call_positional_and_keyword_arguments(forward_statement):
    one = torch.FloatTensor([1])

    model = NeuralNetSimplePositionalAndKeywordArguments()
    pytorch_result = eval(forward_statement + ".item()")

    model = NeuralNetSimplePositionalAndKeywordArguments()
    model = ORTModule(model)
    ortmodule_result = eval(forward_statement + ".item()")
    ortmodule_result_again = eval(forward_statement + ".item()")
    assert ortmodule_result == ortmodule_result_again
    assert pytorch_result == ortmodule_result

    prediction = eval(forward_statement).sum()
    prediction.backward()

def test_torch_nn_module_cuda_method():
    original_device = 'cpu'
    to_device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    model = ORTModule(model)
    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == original_device

    x = torch.randn(N, D_in, device=to_device)
    model = model.cuda()
    model(x)

    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == to_device

@pytest.mark.parametrize("set_gpu_on_original_module", [True, False])
def test_torch_nn_module_cpu_method(set_gpu_on_original_module):
    original_device = 'cuda'
    to_device = 'cpu'

    N, D_in, H, D_out = 64, 784, 500, 10
    if set_gpu_on_original_module:
        model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(original_device)
        model = ORTModule(model)
    else:
        model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
        model = ORTModule(model).to(original_device)
    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == original_device

    x = torch.randn(N, D_in, device=to_device)
    model = model.cpu()
    model(x)
    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == to_device

@pytest.mark.parametrize("original_device", ['cpu', 'cuda'])
@pytest.mark.parametrize("to_argument", ['cpu', 'cuda', 'cuda:0', torch.device('cpu'), torch.device('cuda')])
def test_torch_nn_module_to_api(original_device, to_argument):
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(original_device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=original_device)
    for _, parameter_value in model.named_parameters():
        assert parameter_value.device.type == original_device

    model = model.to(to_argument)
    x = x.to(to_argument)
    model(x)
    assert _utils.get_device_str(model._module_manager._execution_manager(model._is_training())._device) == \
        _utils.get_device_str(torch.device(to_argument))

def test_model_without_device():
    # Model doesn't have device (CPU is assumed)
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    model = ORTModule(model)

    # User input is on GPU
    input_device='cuda'
    x = torch.randn(N, D_in).to(input_device)

    # ORTModule and PyTorch does not move model to where user input is hosted
    with pytest.raises(RuntimeError) as type_error:
        model(x)
    assert \
        ("Tensor for argument #1 'self' is on CPU, but expected them to be on GPU (while checking arguments for addmm)" in str(type_error.value)) \
        or ("Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!" in str(type_error.value))

def test_model_and_input_without_device():
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    model = ORTModule(model)
    x = torch.randn(N, D_in)

    # CPU is assumed for both model and user input
    out = model(x)
    out is not None

def test_model_with_different_devices_same_session():
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    model = ORTModule(model)

    for i in range(5):
        if i % 2 == 0:
            device = 'cpu'
        else:
            device = 'cuda'

        model.to(device)
        x = torch.randn(N, D_in, device=device)
        y = model(x)

@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_input_requires_grad_saved(device):
    N, D_in, H, D_out = 32, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device, requires_grad=True) + 1
    model(x)
    assert model._module_manager._execution_manager(model._is_training())._input_info.require_grad_names == ['input1']

@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_input_requires_grad_backward_creates_input_grad(device):
    N, D_in, H, D_out = 32, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device, requires_grad=True)
    assert x.grad is None
    prediction = model(x)
    s = prediction.sum()
    s.backward()
    assert x.grad is not None

def test_gradient_correctness():
    device = 'cuda'
    N, D_in, H, D_out = 32, 128, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, x):
        prediction = model(x)
        loss = prediction.sum()
        loss.backward()
        return prediction

    for step in range(10):
        x = torch.randn(N, D_in, device=device)
        pt_prediction = run_step(pt_model, x)
        ort_prediction = run_step(ort_model, x)
        
        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)

@pytest.mark.parametrize("use_fp16", [False, True])
@pytest.mark.parametrize("input_requires_grad", [False, True])
def test_gradient_correctness_conv1d(use_fp16, input_requires_grad):
    class NeuralNetConv1D(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, padding=0, groups=1):
            super(NeuralNetConv1D, self).__init__()
            self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, groups=groups)
            self.conv2 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, groups=groups)

        def forward(self, input):
            out = self.conv1(input.permute(0, 2, 1).contiguous())
            out = self.conv2(out).permute(0, 2, 1).contiguous()
            return out

    # ConvGrad hasn't been tested on device with arch lower than 7.0
    if torch.cuda.get_device_capability()[0] < 7:
        return

    device = 'cuda'
    N, seq_len, C_in, C_out, kernel_size = 32, 128, 1536, 1536, 3
    pt_model = NeuralNetConv1D(C_in, C_out, kernel_size, padding=1).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, x):
        with amp.autocast(use_fp16):
            prediction = model(x)
            loss = prediction.sum()
        loss.backward()
        return prediction

    for step in range(10):
        x = torch.randn(N, seq_len, C_in, device=device, requires_grad=input_requires_grad)
        pt_prediction = run_step(pt_model, x)
        ort_prediction = run_step(ort_model, x)
        
        if use_fp16:
            _test_helpers.assert_values_are_close(ort_prediction, pt_prediction, atol=1e-3, rtol=1e-3)
            _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, rtol=1e-2, atol=2e-2)
        else:
            _test_helpers.assert_values_are_close(ort_prediction, pt_prediction, atol=1e-5)
            _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, rtol=5e-3, atol=4e-3)

@pytest.mark.parametrize("device", ['cuda', 'cpu'])
@pytest.mark.parametrize("padding_idx", [None, 1])
def test_gradient_correctness_embedding(device, padding_idx):
    class NeuralNetEmbedding(torch.nn.Module):
        def __init__(self, num_embeddings, embedding_dim, hidden_size):
            super(NeuralNetEmbedding, self).__init__()
            self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
            self.linear = torch.nn.Linear(embedding_dim, hidden_size)

        def forward(self, input):
            return self.linear(self.embedding(input))

    N, num_embeddings, embedding_dim, hidden_size = 64, 32, 128, 128
    pt_model = NeuralNetEmbedding(num_embeddings, embedding_dim, hidden_size).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, input):
        prediction = model(input)
        loss = prediction.sum()
        loss.backward()
        return prediction

    for _ in range(10):
        input = torch.randint(high=num_embeddings, size=(N,), dtype=torch.int64, device=device)
        pt_prediction = run_step(pt_model, input)
        ort_prediction = run_step(ort_model, input)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, atol=1e-5)

@pytest.mark.parametrize("use_fp16", [False, True])
def test_gradient_correctness_cross_entropy_loss(use_fp16):
    class NeuralNetCrossEntropyLoss(torch.nn.Module):
        def __init__(self, num_embeddings, embedding_dim):
            super(NeuralNetCrossEntropyLoss, self).__init__()
            self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=1)

        def forward(self, input, positions):
            output = torch.transpose(self.embedding(input), 0, 1)
            ignored_index = output.size(1)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            return loss_fct(output, positions)

    device = 'cuda'
    num_embeddings, embedding_dim = 32, 128
    pt_model = NeuralNetCrossEntropyLoss(num_embeddings, embedding_dim).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, input, positions):
        with amp.autocast(use_fp16):
            loss = model(input, positions)
        loss.backward()
        return loss

    for _ in range(10):
        N = random.randint(16, 32)
        input = torch.randint(high=num_embeddings, size=(N,), dtype=torch.int64, device=device)
        positions = torch.randint(high=N, size=(embedding_dim,), dtype=torch.int64, device=device)
        pt_prediction = run_step(pt_model, input, positions)
        ort_prediction = run_step(ort_model, input, positions)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, atol=1e-5)

def test_gradient_correctness_maxpool2d():
    class NeuralNetMaxPool2d(torch.nn.Module):
        def __init__(self):
            super(NeuralNetMaxPool2d, self).__init__()
            self.conv = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        def forward(self, input):
            return self.maxpool(self.conv(input))

    N, C, H, W = 8, 3, 224, 224
    device = 'cuda'
    pt_model = NeuralNetMaxPool2d().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, input):
        prediction = model(input)
        loss = prediction.sum()
        loss.backward()
        return prediction

    for _ in range(10):
        input = torch.randn(N, C, H, W, device=device)
        pt_prediction = run_step(pt_model, input)
        ort_prediction = run_step(ort_model, input)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, rtol=5e-3, atol=4e-3)

def test_gradient_correctness_unfold():
    class NeuralNetUnfold(torch.nn.Module):
        def __init__(self, input_size, hidden_size, unfold_dim, unfold_size, unfold_step):
            super(NeuralNetUnfold, self).__init__()
            self.linear= torch.nn.Linear(input_size, hidden_size)
            self.unfold_dim = unfold_dim
            self.unfold_size = unfold_size
            self.unfold_step = unfold_step

        def forward(self, input):
            return self.linear(input).unfold(dimension=self.unfold_dim, size=self.unfold_size, step=self.unfold_step)

    N, D, H = 16, 256, 128
    device = 'cuda'
    pt_model = NeuralNetUnfold(D, H, 1, 50, 30).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, input):
        prediction = model(input)
        loss = prediction.sum()
        loss.backward()
        return prediction

    for _ in range(10):
        input = torch.randn(N, D, device=device)
        pt_prediction = run_step(pt_model, input)
        ort_prediction = run_step(ort_model, input)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)

def test_module_with_non_differential_output():
    device = 'cuda'
    N, D_in, H, D_out = 32, 128, 64, 10
    pt_model = NeuralNetNonDifferentiableOutput(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, x):
        prediction1, mask1, prediction2, mask2 = model(x)
        loss = prediction2.sum()
        loss.backward()
        return prediction1, mask1, prediction2, mask2

    for step in range(10):
        x = torch.randn(N, D_in, device=device)
        pt_prediction1, pt_mask1, pt_prediction2, pt_mask2 = run_step(pt_model, x)
        ort_prediction1, ort_mask1, ort_prediction2, ort_mask2 = run_step(ort_model, x)

        # _test_helpers.assert_values_are_close(ort_prediction1, pt_prediction1)   # TODO: this is failing, need to investigate!
                                                                                   # This will be no reproducible if we change the model forward to 
                                                                                   # mask1 = torch.gt(out, 0.01)
        _test_helpers.assert_values_are_close(ort_prediction2, pt_prediction2)
        _test_helpers.assert_values_are_close(ort_mask1, pt_mask1)
        _test_helpers.assert_values_are_close(ort_mask2, pt_mask2)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)

def test_multiple_forward_only_calls():
    device = 'cuda'
    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    for step in range(10):
        x = torch.randn(N, D_in, device=device, requires_grad=False)
        pt_prediction = pt_model(x)
        ort_prediction = ort_model(x)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)

def test_nesting_forward_backward_calls():
    device = 'cuda'
    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    # forward1
    ort_x1 = torch.randn(N, D_in, device=device, requires_grad=True)
    pt_x1 = copy.deepcopy(ort_x1)
    ort_prediction1 = ort_model(ort_x1)
    pt_prediction1 = pt_model(pt_x1)
    _test_helpers.assert_values_are_close(ort_prediction1, pt_prediction1)
    ort_loss1 = ort_prediction1.sum()
    pt_loss1 = pt_prediction1.sum()
    # forward2
    ort_x2 = torch.randn(N, D_in, device=device, requires_grad=True)
    pt_x2 = copy.deepcopy(ort_x2)
    ort_prediction2 = ort_model(ort_x2)
    ort_loss2 = ort_prediction2.sum()
    pt_prediction2 = pt_model(pt_x2)
    pt_loss2 = pt_prediction2.sum()
    _test_helpers.assert_values_are_close(ort_prediction2, pt_prediction2)
    # backward2
    ort_loss2.backward()
    pt_loss2.backward()
    _test_helpers.assert_values_are_close(ort_x2.grad, ort_x2.grad)
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)
    # backward1
    ort_loss1.backward()
    pt_loss1.backward()
    _test_helpers.assert_values_are_close(ort_x1.grad, pt_x1.grad)
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)

def test_multiple_overlapping_forward_backward_calls():
    device = 'cuda'
    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, x1, x2):
        prediction1 = model(x1)
        loss1 = prediction1.sum()

        prediction2 = model(x2)
        loss2 = prediction2.sum()
        
        loss1.backward()
        loss2.backward()
        return prediction1, prediction2

    for step in range(10):
        pt_x1 = torch.randn(N, D_in, device=device, requires_grad=True)
        pt_x2 = torch.randn(N, D_in, device=device, requires_grad=True)

        ort_x1 = pt_x1.clone().detach()
        ort_x2 = pt_x2.clone().detach()
        ort_x1.requires_grad = True
        ort_x2.requires_grad = True
        
        pt_prediction1, pt_prediction2 = run_step(pt_model, pt_x1, pt_x2)
        ort_prediction1, ort_prediction2 = run_step(ort_model, ort_x1, ort_x2)
        
        _test_helpers.assert_values_are_close(ort_prediction1, pt_prediction1)
        _test_helpers.assert_values_are_close(ort_prediction2, pt_prediction2)
        _test_helpers.assert_values_are_close(ort_x1.grad, pt_x1.grad)
        _test_helpers.assert_values_are_close(ort_x2.grad, pt_x2.grad)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)

def test_multiple_ortmodules_training():
    device = 'cuda'
    N, D_in, H, D_out = 32, 784, 128, 10
    pt_model1 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model2 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model1 = ORTModule(copy.deepcopy(pt_model1))
    ort_model2 = ORTModule(copy.deepcopy(pt_model2))

    def run_step(model1, model2, x1, x2):
        prediction1 = model1(x1)
        loss1 = prediction1.sum()
        loss1.backward()

        prediction2 = model2(x2)
        loss2 = prediction2.sum()
        loss2.backward()
        return prediction1, prediction2

    for step in range(10):
        x1 = torch.randn(N, D_in, device=device)
        x2 = torch.randn(N, D_in, device=device)
        pt_prediction1, pt_prediction2 = run_step(pt_model1, pt_model2, x1, x2)
        ort_prediction1, ort_prediction2 = run_step(ort_model1, ort_model2, x1, x2)

        _test_helpers.assert_values_are_close(ort_prediction1, pt_prediction1, atol=1e-6)
        _test_helpers.assert_values_are_close(ort_prediction2, pt_prediction2, atol=1e-6)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model1, pt_model1)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model2, pt_model2)

def test_multiple_ortmodules_common_backbone_training():
    device = 'cuda'
    N, D_in, H, D_out = 32, 64, 128, 64
    pt_model0 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model1 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model2 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    # model is the common backbone shared by model1 and model2
    ort_model0 = ORTModule(copy.deepcopy(pt_model0))
    ort_model1 = ORTModule(copy.deepcopy(pt_model1))
    ort_model2 = ORTModule(copy.deepcopy(pt_model2))

    def run_step(backbone_layers, task_layers, x):
        prediction = task_layers(backbone_layers(x))
        loss = prediction.sum()
        loss.backward()
        return prediction

    for step in range(10):
        # Run task 1
        x1 = torch.randn(N, D_in, device=device)
        pt_prediction = run_step(pt_model0, pt_model1, x1)
        ort_prediction = run_step(ort_model0, ort_model1, x1)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model0, pt_model0, reset_gradient=False)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model1, pt_model1)

        # Run task 2
        x2 = torch.randn(N, D_in, device=device)
        pt_prediction = run_step(pt_model0, pt_model2, x1)
        ort_prediction = run_step(ort_model0, ort_model2, x1)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model0, pt_model0, reset_gradient=True, atol=1e-5)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model2, pt_model2)

def test_multiple_chained_ortmodules_training():
    device = 'cuda'
    N, D_in, H, D_out = 32, 128, 500, 128
    pt_model1 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model2 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model1 = ORTModule(copy.deepcopy(pt_model1))
    ort_model2 = ORTModule(copy.deepcopy(pt_model2))

    def run_step(layers1, layers2, x):
        prediction = layers2(layers1(x))
        loss = prediction.sum()
        loss.backward()
        return prediction

    for step in range(10):
        x = torch.randn(N, D_in, device=device, requires_grad=True)
        pt_prediction = run_step(pt_model1, pt_model2, x)
        ort_prediction = run_step(ort_model1, ort_model2, x)

        _test_helpers.assert_values_are_close(ort_prediction, pt_prediction)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model1, pt_model1)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model2, pt_model2)

def test_mixed_nnmodule_ortmodules_training():
    device = 'cuda'
    N, D_in, H, D_out = 32, 128, 500, 128
    pt_model1 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model2 = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model3 = NeuralNetMultiplePositionalArguments(D_in, H, D_out).to(device)
    ort_model1 = ORTModule(copy.deepcopy(pt_model1))
    ort_model2 = copy.deepcopy(pt_model2)   # model2 is intentionally left as nn.module
    ort_model3 = ORTModule(copy.deepcopy(pt_model3))

    def run_step(model1, model2, model3, x1, x2):
        a1 = model1(x1)
        a2 = model2(x2)
        a3 = model3(torch.sin(a1), torch.cos(a2))
        loss = a3.sum()
        loss.backward()
        return a1, a2, a3

    for step in range(10):
        x1 = torch.randn(N, D_in, device=device)
        x2 = torch.randn(N, D_in, device=device)
        pt_p1, pt_p2, pt_p3 = run_step(pt_model1, pt_model2, pt_model3, x1, x2)
        ort_p1, ort_p2, ort_p3 = run_step(ort_model1, ort_model2, ort_model3, x1, x2)

        _test_helpers.assert_values_are_close(ort_p1, pt_p1, atol=1e-06)
        _test_helpers.assert_values_are_close(ort_p2, pt_p2, atol=1e-06)
        _test_helpers.assert_values_are_close(ort_p3, pt_p3, atol=1e-06)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model1, pt_model1, atol=2e-6)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model2, pt_model2)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model3, pt_model3)

def test_identity_elimination():
    class NeuralNetSimpleIdentity(torch.nn.Module):
        def __init__(self, input_size, num_classes):
            super(NeuralNetSimpleIdentity, self).__init__()

            self.fc = torch.nn.Linear(input_size, num_classes)

        # Identity node will be created between ReduceSum and graph output
        # and then eliminated after transformation
        def forward(self, x):
            y = self.fc(x)
            z = y 
            return z 

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSimpleIdentity(D_in, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    output = model(x)

    # Make sure model runs OK
    assert output is not None

def test_ortmodule_inputs_with_dynamic_shape():
    D_in, H, D_out = 784, 500, 10

    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to('cuda')
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, x):
        p = model(x)
        loss = p.sum()
        loss.backward()
        return p

    for step in range(10):
        N = random.randint(1,100)
        x = torch.randn(N, D_in, device='cuda', requires_grad=True)
        assert x.grad is None

        pt_p = run_step(pt_model, x)
        ort_p = run_step(ort_model, x)

        _test_helpers.assert_values_are_close(ort_p, pt_p, atol=1e-6)    # relaxing tolerance, 1e-7 or less would fail
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)


def test_bert_inputs_with_dynamic_shape():

    # create pytorch model with dropout disabled
    pt_model = _get_bert_for_sequence_classification_model('cuda', 
        hidden_dropout_prob=0.0, 
        attention_probs_dropout_prob=0.0)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    def run_step(model, x, y, z):
        outputs = model(x, y, None, None, None, None, z)
        loss = outputs[0]
        loss.backward()
        return outputs[0]

    for step in range(10):
        x, y, z = _get_bert_for_sequence_classification_sample_data_with_random_shapes('cuda')

        pt_p = run_step(pt_model, x, y, z)
        ort_p = run_step(ort_model, x, y, z)

        _test_helpers.assert_values_are_close(ort_p, pt_p, atol=1e-02)      # TODO: this assert is failing with smaller tolerance, need to investigate!!
        # _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)  #TODO - enable this check after the investigation


@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_changes_input_requires_grad_reinitializes_module_gradient_graph_builder(device):
    N, D_in, H, D_out = 32, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = x.clone()
    y.requires_grad_(True)
    output_x = torch.sum(model(x))
    output_x.backward()
    assert x.grad is None
    module_gradient_graph_builder_training = \
        model._module_manager._execution_manager(model._is_training())._graph_builder
    output_y = torch.sum(model(y))
    output_y.backward()
    assert y.grad is not None
    assert module_gradient_graph_builder_training != \
        model._module_manager._execution_manager(model._is_training())._graph_builder

@pytest.mark.parametrize("device", ['cuda'])
def test_input_requires_grad_backward_creates_input_grad_as_required0(device):
    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = NeuralNetMultiplePositionalArgumentsMultiOutputsWithoutDependency(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    pt_x1 = torch.randn(N, D_in, device=device, requires_grad=True)
    pt_x2 = torch.randn(N, D_in, device=device, requires_grad=True)
    ort_x1 = pt_x1.clone().detach()
    ort_x2 = pt_x2.clone().detach()
    ort_x1.requires_grad = True
    ort_x2.requires_grad = True

    def run_step0(model, x1, x2):
        y1, _ = model(x1, x2)
        s1 = y1.sum()
        s1.backward()   # y2's gradient will be materialized to full shape.
        return y1

    pt_y1 = run_step0(pt_model, pt_x1, pt_x2)
    ort_y1 = run_step0(ort_model, ort_x1, ort_x2)

    _test_helpers.assert_values_are_close(pt_y1, ort_y1, atol=1e-06)
    _test_helpers.assert_values_are_close(ort_x1.grad, pt_x1.grad)
    _test_helpers.assert_values_are_close(ort_x2.grad, pt_x2.grad)
    # backward() is from y1, so grad of fc2.weight and fc2.bias will not be calculated.
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, none_pt_params=['fc2.weight', 'fc2.bias'], reset_gradient=True)

    def run_step1(model, x1, x2):
        _, y2 = model(x1, x2)
        s2 = y2.sum()
        s2.backward()   # y1's gradient will be materialized to full shape.
        return y2

    pt_y2 = run_step1(pt_model, pt_x1, pt_x2)
    ort_y2 = run_step1(ort_model, ort_x1, ort_x2)

    _test_helpers.assert_values_are_close(pt_y2, ort_y2, atol=1e-06)
    _test_helpers.assert_values_are_close(ort_x1.grad, pt_x1.grad)
    _test_helpers.assert_values_are_close(ort_x2.grad, pt_x2.grad)
    # backward() is from y2, so grad of fc1.weight and fc1.bias will not be calculated.
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model, none_pt_params=['fc1.weight', 'fc1.bias'])


@pytest.mark.parametrize("device", ['cuda'])
def test_model_output_with_inplace_update(device):
    class NeuralNetWithGradNeedOutput(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(NeuralNetWithGradNeedOutput, self).__init__()
            self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
            # Softmax's gradient is depending on its output
            self.act = torch.nn.Softmax(dim=1)

        def forward(self, input1):
            out1 = self.act(self.fc1_1(input1))
            return out1

    def run_step(model, x1):
        y1 = model(x1)
        y1.add_(1)  # inplace update to module output
        y1 = y1.sum()
        y1.backward()
        return y1

    N, D_in, H = 32, 784, 500
    pt_model = NeuralNetWithGradNeedOutput(D_in, H).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    pt_x1 = torch.randn(N, D_in, device=device, requires_grad=True)
    ort_x1 = pt_x1.clone()

    with pytest.raises(Exception) as ex_info:
        pt_y1 = run_step(pt_model, pt_x1)
    assert "modified by an inplace operation" in str(ex_info.value)
    
    with pytest.raises(Exception) as ex_info:
        ort_y1 = run_step(ort_model, ort_x1)
    assert "modified by an inplace operation" in str(ex_info.value)

@pytest.mark.parametrize("device", ['cuda'])
def test_loss_combines_two_outputs_with_dependency(device):

    def run_step(model, x1, x2):
        y1, y2 = model(x1, x2)
        loss = y1.sum() + y2.sum()
        loss.backward()
        return y1, y2

    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = NeuralNetMultiplePositionalArgumentsMultiOutputsWithDependency(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    pt_x1 = torch.randn(N, D_in, device=device, requires_grad=False)
    pt_x2 = torch.randn(N, D_in, device=device, requires_grad=False)
    ort_x1 = pt_x1.clone()
    ort_x2 = pt_x2.clone()

    # Both y1 and y2's gradients are not None.
    pt_y1, pt_y2 = run_step(pt_model, pt_x1, pt_x2)
    ort_y1, ort_y2 = run_step(ort_model, ort_x1, ort_x2)

    _test_helpers.assert_values_are_close(pt_y1, ort_y1, atol=1e-06)
    _test_helpers.assert_values_are_close(pt_y2, ort_y2, atol=1e-06)
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)

@pytest.mark.parametrize("x1_requires_grad", [True, False])
@pytest.mark.parametrize("x2_requires_grad", [True, False])
def test_input_requires_grad_backward_creates_input_grad_as_required1(x1_requires_grad, x2_requires_grad):

    def run_step(model, x1, x2):
        y1, y2 = model(x1, x2)
        s = y2.sum()
        s.backward()
        return y1, y2

    N, D_in, H, D_out = 32, 784, 500, 10
    device = 'cuda'
    pt_model = NeuralNetMultiplePositionalArgumentsMultiOutputsWithDependency(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    pt_x1 = torch.randn(N, D_in, device=device, requires_grad=x1_requires_grad)
    pt_x2 = torch.randn(N, D_in, device=device, requires_grad=x2_requires_grad)

    ort_x1 = pt_x1.clone().detach()
    ort_x2 = pt_x2.clone().detach()
    ort_x1.requires_grad = x1_requires_grad
    ort_x2.requires_grad = x2_requires_grad

    pt_y1, pt_y2 = run_step(pt_model, pt_x1, pt_x2)
    ort_y1, ort_y2 = run_step(ort_model, ort_x1, ort_x2)

    _test_helpers.assert_values_are_close(ort_y1, pt_y1, atol=1e-06)
    _test_helpers.assert_values_are_close(ort_y2, pt_y2, atol=1e-06)
    assert not x1_requires_grad or ort_x1.grad is not None
    assert not x2_requires_grad or ort_x2.grad is not None
    assert not x1_requires_grad or torch.allclose(ort_x1.grad, pt_x1.grad)
    assert not x2_requires_grad or torch.allclose(ort_x2.grad, pt_x2.grad)
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)


@pytest.mark.parametrize("device", ['cuda'])
def test_model_with_bypass_input(device):
    class NeuralNetWithBypassInput(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetWithBypassInput, self).__init__()

            self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1, bypass_input):
            out1 = self.fc1_2(self.relu1(self.fc1_1(input1)))
            # use shape from bypass_input
            out1 = out1.view(bypass_input.size()[0], -1)
            return out1, bypass_input

    def run_step(model, x1, x2):
        y1, y2 = model(x1, x2)
        loss = y1.sum() + y2.sum()
        loss.backward()
        return y1, y2

    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = NeuralNetWithBypassInput(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    pt_x1 = torch.randn(N, D_in, device=device, requires_grad=True)
    pt_x2 = torch.randn(N, D_in, device=device, requires_grad=True)
    ort_x1 = pt_x1.clone()
    ort_x2 = pt_x2.clone()

    # Both y1 and y2's gradients are not None.
    pt_y1, pt_y2 = run_step(pt_model, pt_x1, pt_x2)
    ort_y1, ort_y2 = run_step(ort_model, ort_x1, ort_x2)

    _test_helpers.assert_values_are_close(pt_y1, ort_y1, atol=1e-06)
    _test_helpers.assert_values_are_close(pt_y2, ort_y2, atol=1e-06)
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)

def test_gpu_reserved_memory_with_torch_no_grad():
    device = 'cuda'

    # Create a model and get the memory_reserved when torch.no_grad has been enabled
    # before and after export
    model_with_no_grad = _get_bert_for_sequence_classification_model(device)
    x, y, z = _get_bert_for_sequence_classification_sample_data(device)

    torch.cuda.empty_cache()
    model_with_no_grad = ORTModule(model_with_no_grad)
    model_with_no_grad(x, attention_mask=y, labels=z)
    mem_reserved_after_export_with_torch_no_grad = torch.cuda.memory_reserved(device)
    del model_with_no_grad

    # Create another model and get the memory_reserved when torch.no_grad has not been enabled after export.
    model_without_no_grad = _get_bert_for_sequence_classification_model(device)
    model_without_no_grad = ORTModule(model_without_no_grad)
    mem_reserved_after_export_without_torch_no_grad = 0

    with patch('torch.no_grad'):
        model_without_no_grad(x, attention_mask=y, labels=z)
        mem_reserved_after_export_without_torch_no_grad = torch.cuda.memory_reserved(device)

    assert mem_reserved_after_export_with_torch_no_grad <= mem_reserved_after_export_without_torch_no_grad

@pytest.mark.parametrize("return_type", [dict, OrderedDict, SequenceClassifierOutput])
@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_dict_return_value_module(return_type, device):
    class NeuralNetDictOutput(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetDictOutput, self).__init__()

            self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc2_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu2 = torch.nn.ReLU()
            self.fc2_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc3_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu3 = torch.nn.ReLU()
            self.fc3_2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1, input2, input3):
            out1 = self.fc1_2(self.relu1(self.fc1_1(input1)))
            out2 = self.fc2_2(self.relu2(self.fc2_1(input2)))
            out3 = self.fc3_2(self.relu3(self.fc3_1(input3)))
            return return_type([('loss', out1), ('logits', out2), ('hidden_states', out3)])

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetDictOutput(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    output = model(x, y, z)
    assert isinstance(output, return_type)
    assert 'loss' in output and 'logits' in output and 'hidden_states' in output

@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_dict_of_tuple_return_value_module(device):
    class NeuralNetDictOfTuplesOutput(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetDictOfTuplesOutput, self).__init__()

            self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc2_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu2 = torch.nn.ReLU()
            self.fc2_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc3_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu3 = torch.nn.ReLU()
            self.fc3_2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1, input2, input3):
            out1 = self.fc1_2(self.relu1(self.fc1_1(input1)))
            out2 = self.fc2_2(self.relu2(self.fc2_1(input2)))
            out3 = self.fc3_2(self.relu3(self.fc3_1(input3)))
            return {'loss': (out1, out2, out3)}

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetDictOfTuplesOutput(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    output = model(x, y, z)
    assert 'loss' in output
    assert len(output['loss']) == 3

@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_tuple_of_tuple_return_value_module(device):
    class NeuralNetTupleOfTuplesOutput(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetTupleOfTuplesOutput, self).__init__()

            self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc2_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu2 = torch.nn.ReLU()
            self.fc2_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc3_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu3 = torch.nn.ReLU()
            self.fc3_2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1, input2, input3):
            out1 = self.fc1_2(self.relu1(self.fc1_1(input1)))
            out2 = self.fc2_2(self.relu2(self.fc2_1(input2)))
            out3 = self.fc3_2(self.relu3(self.fc3_1(input3)))
            return ((out1, out2), out3)

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetTupleOfTuplesOutput(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    output = model(x, y, z)
    assert len(output) == 2
    assert isinstance(output[0], tuple)
    assert len(output[0]) == 2
    assert isinstance(output[1], torch.Tensor)

@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_named_tuple_return_value_module(device):
    ReturnValue = namedtuple('NamedTupleReturnValue', 'loss logits hidden_states')
    class NeuralNetNamedTupleOutput(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetNamedTupleOutput, self).__init__()

            self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc2_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu2 = torch.nn.ReLU()
            self.fc2_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc3_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu3 = torch.nn.ReLU()
            self.fc3_2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1, input2, input3):
            out1 = self.fc1_2(self.relu1(self.fc1_1(input1)))
            out2 = self.fc2_2(self.relu2(self.fc2_1(input2)))
            out3 = self.fc3_2(self.relu3(self.fc3_1(input3)))

            return ReturnValue(out1, out2, out3)

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetNamedTupleOutput(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    output = model(x, y, z)
    assert isinstance(output, tuple)
    assert isinstance(output, ReturnValue)

@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_exception_raised_for_custom_class_return_value_module(device):
    class CustomClass(object):
        def __init__(self, out1, out2, out3):
            self.out1 = out1
            self.out2 = out2
            self.out3 = out3

    class NeuralNetCustomClassOutput(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetCustomClassOutput, self).__init__()

            self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc2_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu2 = torch.nn.ReLU()
            self.fc2_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc3_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu3 = torch.nn.ReLU()
            self.fc3_2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1, input2, input3):
            out1 = self.fc1_2(self.relu1(self.fc1_1(input1)))
            out2 = self.fc2_2(self.relu2(self.fc2_1(input2)))
            out3 = self.fc3_2(self.relu3(self.fc3_1(input3)))
            return CustomClass(out1, out2, out3)

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetCustomClassOutput(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    with pytest.raises(TypeError) as runtime_error:
        model(x, y, z)
    assert 'ORTModule does not support the following model output type' in str(runtime_error.value)

def test_dynamic_axes_config():
    device = 'cuda'

    # Model 1
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    output = model(x)
    assert output is not None
    assert _test_helpers.is_dynamic_axes(model)
    del model, output

    # Model 2
    model_with_no_grad = _get_bert_for_sequence_classification_model(device)
    model_with_no_grad = ORTModule(model_with_no_grad)
    x, y, z = _get_bert_for_sequence_classification_sample_data(device)
    output = model_with_no_grad(x, attention_mask=y, labels=z)
    assert output is not None
    assert _test_helpers.is_dynamic_axes(model_with_no_grad)

def test_model_with_multiple_devices_cpu_cuda():
    class MultipleDeviceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 10).cpu()
            self.fc2 = torch.nn.Linear(10, 10).cuda()

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    model = MultipleDeviceModel()
    with pytest.raises(RuntimeError) as e:
        model = ORTModule(model)
    assert str(e.value) == 'ORTModule supports a single device per model for now'

def test_model_with_multiple_devices_to_to():
    class MultipleDeviceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 10).to('cpu')
            self.fc2 = torch.nn.Linear(10, 10).to('cuda')

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    model = MultipleDeviceModel()
    with pytest.raises(RuntimeError) as e:
        model = ORTModule(model)
    assert str(e.value) == 'ORTModule supports a single device per model for now'

def test_model_with_multiple_devices_to_cpu():
    class MultipleDeviceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 10).to('cuda')
            self.fc2 = torch.nn.Linear(10, 10).cpu()

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    model = MultipleDeviceModel()
    with pytest.raises(RuntimeError) as e:
        model = ORTModule(model)
    assert str(e.value) == 'ORTModule supports a single device per model for now'

def test_model_with_multiple_devices_to_cuda():
    class MultipleDeviceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 10).to('cpu')
            self.fc2 = torch.nn.Linear(10, 10).cuda()

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    model = MultipleDeviceModel()
    with pytest.raises(RuntimeError) as e:
        model = ORTModule(model)
    assert str(e.value) == 'ORTModule supports a single device per model for now'

@pytest.mark.parametrize("device", ['cuda', 'cuda:0', 'cuda:1', 'cuda:2'])
def test_model_with_different_cuda_devices(device):

    # Trick to run this test in single GPU machines
    device_id = _utils.get_device_index(device)
    if device_id >= torch.cuda.device_count():
        warnings.warn('Skipping test_model_with_different_cuda_devices(cuda:{})'.format(device_id))
        return

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    model.to(device)
    x = torch.randn(N, D_in, device=device)
    model(x)

def test_register_custom_ops_pytorch_exporter_tensor_triu():
    class SimpleTensorTriuModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 10)

        def forward(self, x):
            x = self.fc1(x)
            mask = torch.ones_like(x).triu(diagonal=1)
            x = x * mask
            return x

    model = SimpleTensorTriuModel()
    model = ORTModule(model)
    user_input = torch.ones(1, 10, 10)

    output = model(user_input)
    assert list(output.shape) ==  [1, 10, 10]

def test_register_custom_ops_pytorch_exporter_torch_triu():
    class SimpleTorchTriuModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 10)

        def forward(self, x):
            x = self.fc1(x)
            mask = torch.triu(torch.ones_like(x))
            x = x * mask
            return x

    model = SimpleTorchTriuModel()
    model = ORTModule(model)
    user_input = torch.ones(1, 10, 10)

    output = model(user_input)
    assert list(output.shape) ==  [1, 10, 10]

def test_wrap_ortmodule_and_change_device():
    # Basic Sequencial model wrapping ORTModule
    x = torch.linspace(-math.pi, math.pi, 2000)
    xx = x.unsqueeze(-1).pow(torch.tensor([1, 2, 3]))
    y = torch.sin(x)
    model = torch.nn.Sequential(
        ORTModule(torch.nn.Linear(3, 1)),
        torch.nn.Flatten(0, 1)
    )

    # Changing device for fun
    model = model.cpu()
    xx = xx.cpu()
    y = y.cpu()
    model = model.cuda()
    xx = xx.cuda()
    y = y.cuda()

    # Quick train
    loss_fn = torch.nn.MSELoss(reduction='sum')
    learning_rate = 1e-6
    for t in range(2000):
        y_pred = model(xx)
        loss = loss_fn(y_pred, y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    # Checking training finished normally
    assert y_pred is not None and loss is not None

@pytest.mark.parametrize("return_dict", [True, False])
def test_hf_model_output_with_tuples(return_dict):
    device = 'cuda'

    model = _get_bert_for_sequence_classification_model(device, output_attentions=True,
        output_hidden_states=True, return_dict=return_dict)
    x, y, z = _get_bert_for_sequence_classification_sample_data(device)

    model = ORTModule(model)
    output = model(x, attention_mask=y, labels=z)

    if return_dict:
        assert isinstance(output, SequenceClassifierOutput)
        assert 'loss' in output and 'logits' in output and \
            'attentions' in output and 'hidden_states' in output
        assert isinstance(output['loss'], torch.Tensor)
        assert isinstance(output['logits'], torch.Tensor)
        assert isinstance(output['attentions'], tuple)
        assert isinstance(output['hidden_states'], tuple)
    else:
        assert isinstance(output, tuple)
        assert isinstance(output[0], torch.Tensor)
        assert isinstance(output[1], torch.Tensor)
        assert isinstance(output[2], tuple)
        assert isinstance(output[3], tuple)

@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_nested_return_value_module(device):
    class NeuralNetNestedOutput(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetNestedOutput, self).__init__()

            self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.relu = torch.nn.ReLU()
            self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc2_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu2 = torch.nn.ReLU()
            self.fc2_2 = torch.nn.Linear(hidden_size, num_classes)

            self.fc3_1 = torch.nn.Linear(input_size, hidden_size)
            self.relu3 = torch.nn.ReLU()
            self.fc3_2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1, input2, input3):
            out1 = self.fc1_2(self.relu1(self.fc1_1(input1)))
            out2 = self.fc2_2(self.relu2(self.fc2_1(input2)))
            out3 = self.fc3_2(self.relu(self.relu3(self.fc3_1(input3))))
            return {
                'a': {
                    'b': {
                        'c': out1
                    },
                    'd': (out2, out3)
                }
            }

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetNestedOutput(D_in, H, D_out).to(device)
    model = ORTModule(model)

    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    output = model(x, y, z)
    assert 'a' in output and 'b' in output['a'] and 'c' in output['a']['b']
    assert isinstance(output['a']['b']['c'], torch.Tensor)

    assert 'd' in output['a']
    assert isinstance(output['a']['d'], tuple)
    assert len(output['a']['d']) == 2

@pytest.mark.parametrize("data_device, model_device", (
    ['cuda', 'cpu'],
    ['cpu', 'cuda'])
)
def test_forward_data_and_model_on_different_devices(data_device, model_device):

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(model_device)
    ort_model = ORTModule(model)
    # When exporting the model, ensure device is same between input data and model (else pytorch will raise while exporting)
    x = torch.randn(N, D_in, device=model_device)
    output = ort_model(x)

    # Now that the model has been exported, feed in data from device other than the model device
    x = torch.randn(N, D_in, device=data_device)
    with pytest.raises(RuntimeError) as runtime_error:
        ort_model(x)
    assert f"Input argument to forward found on device {torch.device(x.device)}, but expected it to be on module device {ort_model._module_manager._execution_manager(ort_model._is_training())._device}." in str(runtime_error.value)

def test_forward_returns_none_type_as_output():
    class NeuralNetNoneTypeOutput(torch.nn.Module):
        def __init__(self, input_size, num_classes):
            super(NeuralNetNoneTypeOutput, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, num_classes)
            self.relu1 = torch.nn.ReLU()

        def forward(self, input1):
            out1 = self.fc1(input1)
            out1 = self.relu1(out1)
            return {'out': out1, 'none_output': None}

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetNoneTypeOutput(D_in, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    output = model(x)

    assert output['out'] is not None
    assert output['none_output'] is None

def test_bool_input_and_output():
    class NeuralNetBoolInputOutput(torch.nn.Module):
        def __init__(self, input_size, num_classes):
            super(NeuralNetBoolInputOutput, self).__init__()
            self.fc = torch.nn.Linear(input_size, num_classes)
            self.relu = torch.nn.ReLU()

        def forward(self, condition, x1, x2):
            out1 = self.relu(self.fc(torch.where(condition, x1, x2)))
            out2 = torch.tensor(out1).to(torch.bool)
            return out1, out2

    device = 'cuda'
    N, D_in, D_out = 64, 784, 10
    model = NeuralNetBoolInputOutput(D_in, D_out).to(device)
    model = ORTModule(model)
    condition = torch.randint(2, (N, D_in), dtype=torch.bool, device=device)
    x1 = torch.randn(N, D_in, device=device)
    x2 = torch.randn(N, D_in, device=device)
    y1, y2 = model(condition, x1, x2)

    assert y1 is not None
    assert y2 is not None and y2.dtype == torch.bool

def test_uint8_input_and_output():
    class NeuralNetUInt8InputOutput(torch.nn.Module):
        def __init__(self, input_size, num_classes):
            super(NeuralNetUInt8InputOutput, self).__init__()
            self.fc = torch.nn.Linear(input_size, num_classes)
            self.relu = torch.nn.ReLU()

        def forward(self, mask, x1, x2):
            out1 = self.relu(self.fc(torch.where(mask == 1, x1, x2)))
            out2 = torch.tensor(out1).to(torch.uint8)
            return out1, out2

    device = 'cuda'
    N, D_in, D_out = 64, 784, 10
    model = NeuralNetUInt8InputOutput(D_in, D_out).to(device)
    model = ORTModule(model)
    condition = torch.randint(2, (N, D_in), dtype=torch.uint8, device=device)
    x1 = torch.randn(N, D_in, device=device)
    x2 = torch.randn(N, D_in, device=device)
    y1, y2 = model(condition, x1, x2)

    assert y1 is not None
    assert y2 is not None and y2.dtype == torch.uint8

def test_model_partially_requires_grad():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetPartialNoGradModel(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)

    # Make sure no exception is raised
    output = model(x)

    loss = torch.sum(output)
    loss.backward()

def test_model_wrapped_inside_torch_no_grad():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)

    # Make sure no exception is raised
    with torch.no_grad():
        output = model(x)

def test_model_initializer_requires_grad_changes_from_one_forward_to_next():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetPartialNoGradModel(D_in, H, D_out).to(device)
    model.fc1.requires_grad_(True)
    model = ORTModule(model)
    x = torch.randn(N, D_in, device=device)
    assert model.module.fc1.weight.grad is None
    assert model.module.fc1.bias.grad is None

    # Make sure no exception is raised
    output = model(x)
    loss = torch.sum(output)
    loss.backward()
    training_session1 = model._module_manager._execution_manager(model._is_training())._execution_agent
    weight_grad_2 = model.module.fc1.weight.grad
    bias_grad_2 = model.module.fc1.bias.grad
    assert weight_grad_2 is not None
    assert bias_grad_2 is not None

    model.module.fc1.requires_grad_(False)
    output = model(x)
    loss = torch.sum(output)
    loss.backward()
    training_session2 = model._module_manager._execution_manager(model._is_training())._execution_agent
    weight_grad_3 = model.module.fc1.weight.grad
    bias_grad_3 = model.module.fc1.bias.grad

    assert training_session1 != training_session2
    assert torch.equal(weight_grad_2, weight_grad_3)
    assert torch.equal(bias_grad_2, bias_grad_3)

def test_model_with_registered_buffers():
    class NeuralNetWithRegisteredBuffer(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetWithRegisteredBuffer, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)
            self.register_buffer("buffer1s", torch.ones(num_classes))
            self.register_buffer("buffer2s", 1+torch.ones(num_classes))

        def forward(self, input1):
            out = self.fc1(input1)
            out = self.relu(out)
            out = self.fc2(out)
            out += self.buffer1s
            out += self.buffer2s
            return out
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetWithRegisteredBuffer(D_in, H, D_out).to(device)
    ort_model = ORTModule(model)
    # Check that the original forward signature is preserved.
    assert signature(model.forward) == signature(ort_model.forward)
    x = torch.randn(N, D_in, device=device)
    # Make sure model runs without any exception
    output = ort_model(x)
    assert output is not None


def test_model_with_unused_registered_buffers():
    class UnusedBufferNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(UnusedBufferNet, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)
            self.register_buffer("buffer1s", torch.ones(num_classes))
            self.register_buffer("buffer2s", 1+torch.ones(num_classes))
            self.register_buffer("buffer3s", 2+torch.ones(num_classes))

        def forward(self, input1):
            out = self.fc1(input1)
            out = self.relu(out)
            out = self.fc2(out)
            out += self.buffer3s
            return out
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = UnusedBufferNet(D_in, H, D_out).to(device)
    ort_model = ORTModule(model)
    # Check that the original forward signature is preserved.
    assert signature(model.forward) == signature(ort_model.forward)
    x = torch.randn(N, D_in, device=device)
    # Make sure model runs without any exception
    output = ort_model(x)
    assert output is not None


def test_model_with_constant_and_registered_parameters():
    class NeuralNetWithRegisteredParamsWithConstant(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetWithRegisteredParamsWithConstant, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)
            self.register_parameter("param1", torch.nn.Parameter(torch.ones(num_classes)))
            self.register_parameter("param2", torch.nn.Parameter(1+torch.ones(num_classes)))

        def forward(self, input1):
            out = self.fc1(input1)
            out = self.relu(out)
            out = self.fc2(out)
            out += self.param1
            out += self.param2
            out += torch.tensor([3.], device=next(self.parameters()).device)
            return out
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetWithRegisteredParamsWithConstant(D_in, H, D_out).to(device)
    ort_model = ORTModule(model)
    # Check that the original forward signature is preserved.
    assert signature(model.forward) == signature(ort_model.forward)
    x = torch.randn(N, D_in, device=device)
    # Make sure model runs without any exception
    output = ort_model(x)
    assert output is not None

def test_state_dict():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    x = torch.randn(N, D_in, device=device)
    y = x.clone()

    state_dict_ort = ort_model.state_dict()
    state_dict_pt = pt_model.state_dict()
    assert state_dict_pt
    assert len(state_dict_pt.keys()) == len(state_dict_ort.keys())
    for param_name, param_value in state_dict_pt.items():
        assert param_name in state_dict_ort
        assert torch.equal(param_value, state_dict_ort[param_name])

    # Call forward once
    ort_model(x)
    pt_model(x)

    state_dict_ort = ort_model.state_dict()
    state_dict_pt = pt_model.state_dict()
    assert state_dict_pt
    assert len(state_dict_pt.keys()) == len(state_dict_ort.keys())
    for param_name, param_value in state_dict_pt.items():
        assert param_name in state_dict_ort
        assert torch.equal(param_value, state_dict_ort[param_name])

def test_load_state_dict():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    x = torch.randn(N, D_in, device=device)
    y = x.clone()

    state_dict_pt = pt_model.state_dict()
    list(next(iter(state_dict_pt.items())))[1] += 10
    ort_model.load_state_dict(state_dict_pt)
    state_dict_ort = ort_model.state_dict()

    assert state_dict_pt
    assert len(state_dict_pt.keys()) == len(state_dict_ort.keys())
    for param_name, param_value in state_dict_pt.items():
        assert param_name in state_dict_ort
        assert torch.equal(param_value, state_dict_ort[param_name])

    # Call forward once
    ort_model(x)
    pt_model(x)

    state_dict_pt = pt_model.state_dict()
    ort_model.load_state_dict(state_dict_pt)
    state_dict_ort = ort_model.state_dict()

    assert state_dict_pt
    assert len(state_dict_pt.keys()) == len(state_dict_ort.keys())
    for param_name, param_value in state_dict_pt.items():
        assert param_name in state_dict_ort
        assert torch.equal(param_value, state_dict_ort[param_name])

def test_named_parameters():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    named_parameters_pt = [name for name, _ in pt_model.named_parameters()]
    named_parameters_ort = [name for name, _ in ort_model.named_parameters()]

    assert len(named_parameters_pt) > 0
    assert named_parameters_pt == named_parameters_ort

def test_parameters():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    parameters_pt = [param for param in pt_model.parameters()]
    parameters_ort = [param for param in ort_model.parameters()]

    assert len(parameters_pt) > 0
    assert len(parameters_pt) == len(parameters_ort)
    assert all(torch.equal(parameters_pt[i], parameters_ort[i]) for i in range(len(parameters_pt)))

def test_named_buffers():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model.register_buffer('sample_buffer_pt', torch.tensor(torch.randn(N, D_in, device=device)))
    ort_model = ORTModule(copy.deepcopy(pt_model))
    named_buffers_pt = [name for name, _ in pt_model.named_buffers()]
    named_buffers_ort = [name for name, _ in ort_model.named_buffers()]

    assert len(named_buffers_pt) > 0
    assert named_buffers_pt == named_buffers_ort

    ort_model.register_buffer('sample_buffer_ort', torch.tensor(torch.randn(N, D_in, device=device)))
    named_buffers_ort = [name for name, _ in ort_model.named_buffers()]
    assert named_buffers_ort == ['sample_buffer_pt', 'sample_buffer_ort']

def test_buffers():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model.register_buffer('sample_buffer_pt', torch.tensor(torch.randn(N, D_in, device=device)))
    ort_model = ORTModule(copy.deepcopy(pt_model))
    buffers_pt = [buffer for buffer in pt_model.buffers()]
    buffers_ort = [buffer for buffer in ort_model.buffers()]

    assert len(buffers_pt) > 0
    assert len(buffers_pt) == len(buffers_ort)
    assert all(torch.equal(buffers_pt[i], buffers_ort[i]) for i in range(len(buffers_pt)))

    x = torch.tensor(torch.randn(N, D_in, device=device))
    ort_model.register_buffer('sample_buffer_ort', x)
    buffers_ort = [buffer for buffer in ort_model.buffers()]
    assert len(buffers_ort) == 2
    assert torch.equal(buffers_ort[1], x)

def test_eval_with_dropout():
    class NeuralNetDropout(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetDropout, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)
            self.dropout = torch.nn.Dropout()

        def forward(self, input1):
            out = self.fc1(input1)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.dropout(out)
            return out

    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetDropout(D_in, H, D_out).to(device)
    model.eval()
    ort_model = ORTModule(copy.deepcopy(model))
    ort_model.eval()

    x = torch.randn(N, D_in, device=device)
    y = x.clone()

    # Make sure model runs without any exception
    output = ort_model(x)
    output_pt = model(y)

    assert output is not None
    assert output_pt is not None
    # Assert that the output from torch is the same as the one from ORTModule
    _test_helpers.assert_values_are_close(output, output_pt)

def test_with_torch_no_grad_context():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(model))

    x = torch.randn(N, D_in, device=device)
    y = x.clone()

    # Make sure model runs without any exception
    output = None
    output_pt = None
    with torch.no_grad():
        output = ort_model(x)
        output_pt = model(y)

    assert output is not None
    assert output_pt is not None
    # Assert that the output from torch is the same as the one from ORTModule
    _test_helpers.assert_values_are_close(output, output_pt)
    assert output.grad is None and output_pt.grad is None

def test_unused_layer():
    class Net(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(Net, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1):
            out = self.fc1(input1)
            out = self.relu(out)
            return out

    device = torch.device('cuda')
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    ort_model = ORTModule(model)

    x = torch.randn(N, D_in, device=device)
    output = ort_model(x)
    assert output is not None

def test_forward_dynamic_args():
    device = 'cuda'

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetPositionalArguments(input_size=D_in, hidden_size=H, num_classes=D_out).to(device)
    model = ORTModule(model)
    args_size1 = [torch.randn(N, D_in, device=device)]*4
    args_size2 = [torch.randn(N, D_in, device=device)]*3
    args_size3 = [torch.randn(N, D_in, device=device)]*5

    # Make sure model runs without any exception
    for i in range(2):

        # Test both train and inference mode
        if i % 2 == 0:
            model.train()
        else:
            model.eval()

        # Train model with one set of input
        for _ in range(10):
            output = model(*args_size1)
            assert output is not None
        hash_args_size1 = hash(repr(model._module_manager._execution_manager(model._is_training())._input_info.schema))
        assert hash_args_size1 is not None

        # Decrease number of inputs and train some more
        for _ in range(10):
            output = model(*args_size2)
            assert output is not None
        hash_args_size2 = hash(repr(model._module_manager._execution_manager(model._is_training())._input_info.schema))
        assert hash_args_size2 != hash_args_size1

        # Increase number of inputs and train some more
        for _ in range(10):
            output = model(*args_size3)
            assert output is not None
        hash_args_size3 = hash(repr(model._module_manager._execution_manager(model._is_training())._input_info.schema))
        assert hash_args_size3 != hash_args_size2


def test_forward_dynamic_kwargs():
    one = torch.FloatTensor([1])
    model = NeuralNetSimplePositionalAndKeywordArguments()
    model = ORTModule(model)

    # Make sure model runs without any exception
    for i in range(2):

        # Test both train and inference mode
        if i % 2 == 0:
            model.train()
        else:
            model.eval()

        # Train model with positional argument x only
        for _ in range(10):
            output = model(one)
            assert output is not None
        hash_x = hash(repr(model._module_manager._execution_manager(model._is_training())._input_info.schema))
        assert hash_x is not None

        # Train with x and y as inputs
        for _ in range(10):
            output = model(one,y=one)
            assert output is not None
        hash_x_y = hash(repr(model._module_manager._execution_manager(model._is_training())._input_info.schema))
        assert hash_x_y != hash_x

        # Train with x and z as inputs
        for _ in range(10):
            output = model(one,z=one)
            assert output is not None
        hash_x_z = hash(repr(model._module_manager._execution_manager(model._is_training())._input_info.schema))
        assert hash_x_z != hash_x_y

        # Train with x, y and z as inputs
        for _ in range(10):
            output = model(one,y=one, z=one)
            assert output is not None
        hash_x_y_z = hash(repr(model._module_manager._execution_manager(model._is_training())._input_info.schema))
        assert hash_x_y_z != hash_x_z

        # Return to original input with x as input
        for _ in range(10):
            output = model(one)
            assert output is not None
        hash_x2 = hash(repr(model._module_manager._execution_manager(model._is_training())._input_info.schema))
        assert hash_x2 != hash_x_y_z
        assert hash_x2 == hash_x


@pytest.mark.parametrize("forward_statement",
                         [# Only pos_X, pos_X as positionals
                          "model(pos_0, pos_1)",
                          # Only pos_X, pos_X as keywords
                          "model(pos_0=pos_0, pos_1=pos_1)",
                          # pos_X + *args, pos_X as positionals
                          "model(pos_0, pos_1, *args)",
                          # pos_X + kw_X, pos_X as positionals
                          "model(pos_0, pos_1, kw_0=kw_0, kw_1=kw_1)",
                          # pos_X + kw_X,  pos_X as keywords
                          "model(pos_0=pos_0, pos_1=pos_1, kw_0=kw_0, kw_1=kw_1)",
                          # pos_X + kw_X, pos_X as positionals (missing kw_1)
                          "model(pos_0, pos_1, kw_0=kw_0)",
                          # pos_X + kw_X, pos_X as keywords (missing kw_1)
                          "model(pos_0=pos_0, pos_1=pos_1, kw_0=kw_0)",
                          # pos_X + kw_X, pos_X as positionals (missing kw_0)
                          "model(pos_0, pos_1, kw_1=kw_1)",
                          # pos_X + kw_X, pos_X as keywords (missing kw_0)
                          "model(pos_0=pos_0, pos_1=pos_1, kw_1=kw_1)",
                          # pos_X + kwargs, pos_X as positionals
                          "model(pos_0, pos_1, **kwargs)",
                          # pos_X + kwargs, pos_X as keywords
                          "model(pos_0=pos_0, pos_1=pos_1, **kwargs)",
                          # pos_X + *args + kw_X, pos_X as positionals
                          "model(pos_0, pos_1, *args, kw_0=kw_0, kw_1=kw_1)",
                          # pos_X + *args + kw_X, pos_X as positionals (missing kw_0)
                          "model(pos_0, pos_1, *args, kw_1=kw_1)",
                          # pos_X + *args + kw_X, pos_X as positionals (missing kw_1)
                          "model(pos_0, pos_1, *args, kw_0=kw_0)",
                          # pos_X + *args + kwargs, pos_X as positionals
                          "model(pos_0, pos_1, *args, **kwargs)",
                          # pos_X + *args + kw_X + kwargs, pos_X as positionals
                          "model(pos_0, pos_1, *args, kw_0=kw_0, kw_1=kw_1, **kwargs)",
                          # pos_X + *args + kw_X + kwargs, pos_X as positionals (missing kw_0)
                          "model(pos_0, pos_1, *args, kw_1=kw_1, **kwargs)",
                          # pos_X + *args + kw_X + kwargs, pos_X as positionals (missing kw_1)
                          "model(pos_0, pos_1, *args, kw_0=kw_0, **kwargs)",
                          ])
def test_forward_call_kwargs_input(forward_statement):
    class KwargsNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(KwargsNet, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, pos_0, pos_1, *args, kw_0=None, kw_1=None, **kwargs):
            model_input = pos_0 + pos_1
            if args:
                model_input += sum(args)
            if kw_0 is not None:
                model_input += kw_0
            if kw_1 is not None:
                model_input += kw_1
            if kwargs:
                if 'kwargs_0' in kwargs:
                    model_input += kwargs['kwargs_0']
                if 'kwargs_1' in kwargs:
                    model_input += torch.matmul(kwargs['kwargs_0'], kwargs['kwargs_1'])

            out = self.fc1(model_input)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    # Modeling
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = KwargsNet(input_size=D_in, hidden_size=H, num_classes=D_out).to(device)
    model = ORTModule(model)

    # Dummy inputs used
    pos_0 = torch.randn(N, D_in, device=device)
    pos_1 = torch.randn(N, D_in, device=device)
    kw_0 = torch.randn(N, D_in, device=device)
    kw_1 = torch.randn(N, D_in, device=device)
    args = [torch.randn(N, D_in, device=device)]*2
    kwargs = {'kwargs_0' : torch.randn(N, D_in, device=device),
              'kwargs_1' : torch.randn(D_in, D_in, device=device)}

    # Training step
    prediction = eval(forward_statement)
    assert prediction is not None
    prediction = prediction.sum()
    prediction.backward()


def test_repro_iscontiguous():
    class SimpleNet(torch.nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.a = torch.nn.Parameter(torch.FloatTensor([-1., 1.]))
        def forward(self, x):
            result = torch.mean(self.a) + x
            return result

    one = torch.FloatTensor([1])
    model = SimpleNet()
    model = ORTModule(model)
    prediction = model(one)
    prediction = prediction.sum()
    prediction.backward()


def test_forward_call_default_input():
    class UnusedNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.zeros = torch.nn.Parameter(torch.zeros(1,1))

        def forward(self, a, b, c, d, *args, kw_0=None, **kwargs):
            result = a + d + self.zeros.sum()
            if args:
                result += args[-1]
            if kw_0:
                result += kw_0
            if kwargs:
                assert 'kwargs_1' in kwargs
                result += kwargs['kwargs_1']
            return result

    # Modeling
    device = 'cuda'
    model = UnusedNet().to(device)
    model = ORTModule(model)

    # Dummy data
    one = torch.FloatTensor([1]).to(device)
    two = 2*one
    three = 3*one
    four = 4*one
    args = [two]*5
    kw_0 = 6*one
    kwargs = {'kwargs_0': 7*one, 'kwargs_1': 8*one}

    # Make sure model runs without any exception
    for i in range(2):
        # Test both train and inference mode
        if i % 2 == 0:
            model.train()
        else:
            model.eval()

        # Model only uses a,d out of a,b,c,d
        out = model(one, two, three, four)
        assert out.item() == 5.0
        if model._is_training():
            out.sum().backward()

        out = model(one, two, c=three, d=four)
        assert out.item() == 5.0
        if model._is_training():
            out.sum().backward()

        # Model only uses a,d,args[-1] out of a,b,c,d,*args
        out = model(one, two, three, four, *args)
        assert out.item() == 7.0
        if model._is_training():
            out.sum().backward()

        # Model only uses a,d,args[-1],kw_0 out of a,b,c,d,*args,kw_0
        out = model(one, two, three, four, *args, kw_0=kw_0)
        assert out.item() == 13.0
        if model._is_training():
            out.sum().backward()

        # Model only uses a,d,args[-1],kwargs['kwargs_1'] out of a,b,c,d,*args,kw_0,**kwargs
        out = model(one, two, three, four, *args, **kwargs)
        assert out.item() == 15.0
        if model._is_training():
            out.sum().backward()


def test_forward_call_kwargs_input_unexpected_order():
    class OrderlyNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(OrderlyNet, self).__init__()
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1=None, input2=None):
            assert input1.shape != input2.shape
            input2 = torch.transpose(input2, 0, 1)
            assert input1.shape == input2.shape

            model_input = input1 + input2
            out1 = self.fc1(model_input)
            out1 = self.relu(out1)
            out2 = self.fc2(out1)
            return out1, out2

    device = 'cuda'
    N, D_in, H, D_out = 32, 784, 500, 10
    model = OrderlyNet(D_in, H, D_out).to(device)
    model = ORTModule(model)

    input1 = torch.randn(N, D_in, device=device, requires_grad=False)
    input2 = torch.randn(D_in, N, device=device, requires_grad=False)

    # Make sure model runs without any exception
    for i in range(2):
        # Test both train and inference mode
        if i % 2 == 0:
            model.train()
        else:
            model.eval()

        # Must work because forward() and dict order match
        y1, y2 = model(**{'input1': input1, 'input2': input2})
        assert y1 is not None
        assert y2 is not None
        if model._is_training():
            loss = y1.sum() + y2.sum()
            loss.backward()

        # Must work even when forward() and dict order mismatch
        y1, y2 = model(**{'input2': input2, 'input1': input1})
        assert y1 is not None
        assert y2 is not None
        if model._is_training():
            loss = y1.sum() + y2.sum()
            loss.backward()


def test_forward_call_lots_None():
    class NoneNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.zeros = torch.nn.Parameter(torch.zeros(1,1))

        def forward(self, a, b, c, d, e, f, y=None, z=None):
            assert a is not None
            result = self.zeros.sum() + a
            if b is not None:
                result += b
            if c is not None:
                result += c
            if d is not None:
                result += d
            if e is not None:
                result += e
            if f is not None:
                result += f
            if y is not None:
                result += y
            if z is not None:
                result += z
            return result

    def run_step(expected, a, b, c, d, e, f, y, z):
        # Force model (re)export to validate (un)flattening with new input
        #   This is needed because for a `forward(self, a, b)`, and
        #   input `forward(a,b)` or `forward(**{'a': a, 'b': b})`,
        #   ORTModule produces the same schema, thus not re-exporting
        #   the model when `forward(a,b)` is used after `forward(**{'a': a, 'b': b})`
        #   or vice-versa
        model._module_manager._execution_manager(model._is_training())._onnx_model = None
        out = model(a,b,c,d,e,f,y,z)
        assert out is not None
        assert out.item() == expected
        if model._is_training():
            loss = out.sum()
            loss.backward()

    device = 'cuda'
    model = NoneNet().to(device)
    model = ORTModule(model)

    a = torch.FloatTensor([1]).to(device)*1
    b = torch.FloatTensor([1]).to(device)*10
    c = torch.FloatTensor([1]).to(device)*100
    d = torch.FloatTensor([1]).to(device)*1000
    e = torch.FloatTensor([1]).to(device)*10000
    f = torch.FloatTensor([1]).to(device)*100000
    y = torch.FloatTensor([1]).to(device)*1000000
    z = torch.FloatTensor([1]).to(device)*10000000

    # Make sure model runs without any exception
    for i in range(2):
        # Test both train and inference mode
        if i % 2 == 0:
            model.train()
        else:
            model.eval()

        run_step(a.item() + f.item(),
                 a, None, None, None, None, f, None, None, )
        run_step(a.item() + f.item(),
                 **{'a': a, 'b': None, 'c': None, 'd': None, 'e': None, 'f': f, 'y': None, 'z': None})
        run_step(a.item() + z.item(),
                 a, None, None, None, None, None, None, z)
        run_step(a.item() + z.item(),
                 **{'a': a, 'b': None, 'c': None, 'd': None, 'e': None, 'f': None, 'y': None, 'z': z})
        run_step(a.item() + c.item() + y.item(),
                 a, None, c, None, None, None, y, None)
        run_step(a.item() + c.item() + y.item(),
                 **{'a': a, 'b': None, 'c': c, 'd': None, 'e': None, 'f': None, 'y': y, 'z': None})
        run_step(a.item() + b.item() + c.item() + d.item() + e.item() + f.item() + y.item() + z.item(),
                 a, b, c, d, e, f, y, z)
        run_step(a.item() + b.item() + c.item() + d.item() + e.item() + f.item() + y.item() + z.item(),
                 **{'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f, 'y': y, 'z': z})

@pytest.mark.parametrize("bool_argument", [True, False])
@pytest.mark.parametrize("int_argument", [100, 100000, 100000000, -100, -100000, -100000000])
@pytest.mark.parametrize("float_argument", [1.23, 11209123.12452, 12093702935.1249863, -1.23, -11209123.12452, -12093702935.1249863])
def test_primitive_inputs(bool_argument, int_argument, float_argument):
    class PrimitiveTypesInputNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(PrimitiveTypesInputNet, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1, bool_argument, int_argument, float_argument):
            input1 = input1 + int_argument + float_argument
            if bool_argument:
                out = self.fc1(input1)
                out = self.relu(out)
                out = self.fc2(out)
            else:
                out = self.fc1(input1)
                out = self.fc2(out)
                out = self.relu(out)
            return out

    assert type(bool_argument) is bool
    assert type(int_argument) is int
    assert type(float_argument) is float

    device = 'cuda'
    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = PrimitiveTypesInputNet(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    input1 = torch.randn(N, D_in, device=device)
    pt_out = pt_model(input1, bool_argument, int_argument, float_argument)
    ort_out = ort_model(input1, bool_argument, int_argument, float_argument)
    _test_helpers.assert_values_are_close(pt_out, ort_out)

@pytest.mark.parametrize("bool_arguments", [(True, False), (False, True)])
def test_changing_bool_input_re_exports_model(bool_arguments):
    class PrimitiveTypesInputNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(PrimitiveTypesInputNet, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input1, bool_argument):
            if bool_argument:
                out = self.fc1(input1)
                out = self.relu(out)
                out = self.fc2(out)
            else:
                out = self.fc1(input1)
                out = self.fc2(out)
                out = self.relu(out)
            return out

    assert type(bool_arguments[0]) is bool
    assert type(bool_arguments[1]) is bool

    device = 'cuda'
    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = PrimitiveTypesInputNet(D_in, H, D_out).to(device)
    ort_model = ORTModule(pt_model)

    input1 = torch.randn(N, D_in, device=device)
    ort_model(input1, bool_arguments[0])
    exported_model1 = ort_model._module_manager._execution_manager(ort_model._is_training())._onnx_model

    ort_model(input1, bool_arguments[1])
    exported_model2 = ort_model._module_manager._execution_manager(ort_model._is_training())._onnx_model

    assert exported_model1 != exported_model2

def test_model_with_registered_buffer_and_dropped_parameters():
    class ModelWithBufferAndDroppedParameter(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(ModelWithBufferAndDroppedParameter, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)
            self.register_buffer("buffer", torch.ones(num_classes))

        def forward(self, bool_argument, input1):
            if bool_argument:
                out = self.fc1(input1)
                out = self.relu(out)
                out = self.fc2(out)
                out = out + self.buffer
            else:
                out = self.fc1(input1)
                out = self.fc2(out)
                out = self.relu(out)
                out = out + self.buffer
            return out

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = ModelWithBufferAndDroppedParameter(D_in, H, D_out).to(device)
    model = ORTModule(model)

    bool_argument = torch.tensor(True)
    x = torch.randn(N, D_in, device=device)

    # Ensure that no exceptions are raised
    out = model(bool_argument, x)

@pytest.mark.parametrize("model, none_pt_params",
        [(UnusedBeginParameterNet(784, 500, 400, 10), ['fc1.weight', 'fc1.bias']),
         (UnusedMiddleParameterNet(784, 500, 400, 10), ['fc2.weight', 'fc2.bias']),
         (UnusedEndParameterNet(784, 500, 400, 10), ['fc2.weight', 'fc2.bias'])])
def test_unused_parameters(model, none_pt_params):
    device = 'cuda'

    N, D_in, H1, H2, D_out = 64, 784, 500, 400, 10
    model = model.to(device)
    ort_model = ORTModule(copy.deepcopy(model))

    # Make sure model runs without any exception
    for _ in range(5):
        x = torch.randn(N, D_in, device=device)
        y = copy.deepcopy(x)

        out_pt = model(x)
        out_ort = ort_model(y)
        loss_pt = out_pt.sum()
        loss_pt.backward()
        loss_ort = out_ort.sum()
        loss_ort.backward()
        _test_helpers.assert_values_are_close(out_ort, out_pt)
        _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, model,
            none_pt_params=none_pt_params)

    # Also try in eval mode
    model.eval()
    ort_model.eval()

    x = torch.randn(N, D_in, device=device)
    y = copy.deepcopy(x)

    # Make sure model runs without any exception
    out_pt = model(x)
    out_ort = ort_model(y)
    _test_helpers.assert_values_are_close(out_ort, out_pt)

def test_output_order():
    class OutputOrderNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(OutputOrderNet, self).__init__()

            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.fc2 = torch.nn.Linear(input_size, hidden_size)
            self.fc3 = torch.nn.Linear(input_size, hidden_size)
            self.fc4 = torch.nn.Linear(input_size, hidden_size)
            self.fc5 = torch.nn.Linear(input_size, hidden_size)
            self.fc6 = torch.nn.Linear(input_size, hidden_size)
            self.fc7 = torch.nn.Linear(input_size, hidden_size)
            self.fc8 = torch.nn.Linear(input_size, hidden_size)
            self.fc9 = torch.nn.Linear(input_size, hidden_size)
            self.fc10 = torch.nn.Linear(input_size, hidden_size)
            self.fc11 = torch.nn.Linear(input_size, hidden_size)
            self.fc12 = torch.nn.Linear(input_size, hidden_size)

        def forward(self, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12):
            return self.fc1(input1), self.fc2(input2), self.fc3(input3), \
                self.fc4(input4), self.fc5(input5), self.fc6(input6), \
                self.fc7(input7), self.fc8(input8), self.fc9(input9), \
                self.fc10(input10), self.fc11(input11), self.fc12(input12)

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = OutputOrderNet(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(model))

    x = [torch.randn(N, D_in, device=device) for _ in range(12)]
    y = copy.deepcopy(x)

    out_pt = model(*x)
    out_ort = ort_model(*y)

    assert len(out_pt) == len(out_ort)
    for x, y in zip(out_pt, out_ort):
        _test_helpers.assert_values_are_close(x, y)

@pytest.mark.parametrize("device", ['cuda', 'cpu', None])
def test_stateless_model_specified_device(device):

    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = StatelessModel().to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))

    pt_x = torch.randn(N, D_in, device=device)
    ort_x = pt_x.clone()

    pt_y = pt_model(pt_x)
    ort_y = ort_model(ort_x)

    _test_helpers.assert_values_are_close(pt_y, ort_y)

def test_stateless_model_unspecified_device():

    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = StatelessModel()
    ort_model = ORTModule(copy.deepcopy(pt_model))

    pt_x = torch.randn(N, D_in)
    ort_x = pt_x.clone()

    pt_y = pt_model(pt_x)
    ort_y = ort_model(ort_x)

    _test_helpers.assert_values_are_close(pt_y, ort_y)

@pytest.mark.parametrize("model",
        [(UnusedBeginParameterNet(784, 500, 400, 10)),
         (UnusedMiddleParameterNet(784, 500, 400, 10)),
         (UnusedEndParameterNet(784, 500, 400, 10))])
def test_unused_parameters_does_not_unnecssarily_reinitilize(model):
    device = 'cuda'

    N, D_in, H1, H2, D_out = 64, 784, 500, 400, 10
    model = model.to(device)
    ort_model = ORTModule(copy.deepcopy(model))
    training_manager = ort_model._module_manager._execution_manager(ort_model._is_training())

    x = torch.randn(N, D_in, device=device)
    _ = ort_model(x)

    input_info = _io.parse_inputs_for_onnx_export(training_manager._module_parameters,
                                                  training_manager._onnx_model,
                                                  x,
                                                  {})

    assert not training_manager._reinitialize_graph_builder(input_info)

def test_load_state_dict_for_wrapped_ortmodule():
    class WrapperModule(torch.nn.Module):
        def __init__(self, ortmodule):
            super(WrapperModule, self).__init__()
            self._ortmodule = ortmodule

        def forward(self, x):
            return self._ortmodule(x)

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(copy.deepcopy(model))
    wrapper_module = WrapperModule(model)
    x = torch.randn(N, D_in, device=device)
    _ = wrapper_module(x)

    state_dict1 = wrapper_module.state_dict()
    list(next(iter(state_dict1.items())))[1] += 10
    wrapper_module.load_state_dict(state_dict1)
    state_dict2 = wrapper_module.state_dict()

    assert state_dict1
    assert len(state_dict1.keys()) == len(state_dict2.keys())
    for param_name, param_value in state_dict1.items():
        assert param_name in state_dict2
        assert torch.equal(param_value, state_dict2[param_name])

def test_hf_save_pretrained():
    device = 'cuda'

    model1 = _get_bert_for_sequence_classification_model(device)
    model1 = ORTModule(model1)
    state_dict = model1.state_dict()
    list(next(iter(state_dict.items())))[1] += 100
    model1.load_state_dict(state_dict)

    trainer = Trainer(model=model1)

    # Assert that ORTModule has an attribute called module. This attribute is used
    # for trainer.save_model to reference the underlying HuggingFace PreTrainedModel
    assert hasattr(model1, "module")

    # Create a temporary directory for the checkpoint from save_pretrained
    with tempfile.TemporaryDirectory() as temporary_dir:
        trainer.save_model(temporary_dir)

        # Create a new model and compare all state dictionary values for equality
        # to check if from_pretrained worked.
        config = AutoConfig.from_pretrained(temporary_dir)
        model2 = BertForSequenceClassification.from_pretrained(
            temporary_dir, config=config,
        ).to(device)
        model2 = ORTModule(model2)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert p1.data.ne(p2.data).sum() == 0

def test_input_with_string_exception():
    class MyStrNet(torch.nn.Module):
        def forward(self, x, my_str):
            if my_str.lower() == 'hello':
                print('hi')
            return x

    model = MyStrNet()
    model = ORTModule(model)
    with pytest.raises(TypeError) as ex_info:
        _ = model(torch.randn(1, 2), 'hello')
    assert "ORTModule does not support the following model data type <class 'str'>" in str(ex_info.value)
