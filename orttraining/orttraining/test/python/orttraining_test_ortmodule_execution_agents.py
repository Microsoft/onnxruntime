from onnx import helper, TensorProto
import pytest
import torch

import onnxruntime
from onnxruntime.training.ortmodule._execution_agent import (
    InferenceAgent,
    TrainingAgent,
)
from _test_helpers import assert_values_are_close # pylint: disable=wrong-import-order


@pytest.fixture
def inference_model():
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3]),
        helper.make_tensor_value_info("bias", TensorProto.FLOAT, [3]),
    ]
    mul_node = helper.make_node("Mul", ["x", "y"], ["mul"])
    add_node = helper.make_node("Add", ["mul", "bias"], ["add"])
    outputs = [helper.make_tensor_value_info("add", TensorProto.FLOAT, [2, 3])]
    graph_def = helper.make_graph([mul_node, add_node], "graph", inputs, outputs)
    return helper.make_model(graph_def)


@pytest.fixture
def training_model():
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3]),
        helper.make_tensor_value_info("bias", TensorProto.FLOAT, [3]),
    ]
    mul_node = helper.make_node("Mul", ["x", "y"], ["mul"])
    add_node = helper.make_node("Add", ["mul", "bias"], ["add"])
    yield_node = helper.make_node(
        "YieldOp", ["add"], ["add_grad"], full_shape_outputs=[0], domain="com.microsoft"
    )
    x_grad_node = helper.make_node("Mul", ["add_grad", "y"], ["x_grad"])
    y_grad_node = helper.make_node("Mul", ["x", "add_grad"], ["y_grad"])
    axes_tensor = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    bias_grad_node = helper.make_node(
        "ReduceSum", ["add_grad", "axes"], ["bias_grad"], keepdims=0
    )
    outputs = [
        helper.make_tensor_value_info("x_grad", TensorProto.FLOAT, [2, 3]),
        helper.make_tensor_value_info("y_grad", TensorProto.FLOAT, [2, 3]),
        helper.make_tensor_value_info("bias_grad", TensorProto.FLOAT, [3]),
    ]
    graph_def = helper.make_graph(
        [mul_node, add_node, yield_node, x_grad_node, y_grad_node, bias_grad_node],
        "graph",
        inputs,
        outputs,
        initializer=[axes_tensor],
    )
    return helper.make_model(graph_def)


@pytest.fixture
def execution_agent_kwargs():
    return dict(
        session_options=onnxruntime.SessionOptions(),
        providers=[("CPUExecutionProvider", {})],
    )


@pytest.fixture
def inference_agent(inference_model, execution_agent_kwargs):
    return InferenceAgent(inference_model, torch.device("cpu"), **execution_agent_kwargs)


@pytest.fixture
def training_agent(training_model, execution_agent_kwargs):
    return TrainingAgent(training_model, torch.device("cpu"), **execution_agent_kwargs)


def test_onnx_graph_forward(inference_agent):
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    bias = torch.randn(3)
    expected_output = x * y + bias
    outputs, _ = inference_agent.forward(x, y, bias)
    assert_values_are_close(outputs[0], expected_output)


def test_onnx_graph_forward_backward(training_agent):
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    bias = torch.randn(3)
    x.requires_grad = y.requires_grad = bias.requires_grad = True
    expected_output = x * y + bias
    add_grad = torch.randn(2, 3)
    expected_output.backward(add_grad)
    expected_grads = [x.grad.clone(), y.grad.clone(), bias.grad.clone()]
    x.grad = y.grad = bias.grad = None
    outputs, run_info = training_agent.forward(x, y, bias)
    grads = training_agent.backward(run_info, add_grad)
    assert_values_are_close(outputs[0], expected_output)
    for grad, expected_grad in zip(grads, expected_grads):
        assert_values_are_close(grad, expected_grad)
