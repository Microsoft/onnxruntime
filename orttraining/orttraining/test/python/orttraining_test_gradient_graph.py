import os
import unittest
from pathlib import Path

import torch

from onnxruntime.training import GradientGraphBuilder


class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class GradientGraphBuilderTest(unittest.TestCase):
    def test_save(self):
        model = NeuralNet(input_size=10, hidden_size=5, num_classes=2)
        directory_path = Path(os.path.dirname(__file__)).resolve()
        path = directory_path / 'model.onnx'
        batch_size = 1
        x = torch.randn(batch_size, model.fc1.in_features, requires_grad=True)
        torch.onnx.export(
            model, x, str(path),
            export_params=True,
            opset_version=12, do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', },
                'output': {0: 'batch_size', },
            })
        # FIXME Gets a Segmentation fault.
        builder = GradientGraphBuilder(str(path), {'output'}, {'input'}, 'loss')
        builder.build()
        # TODO Maybe it should be .ort?
        gradient_graph_path = directory_path/'gradient_graph_model.onnx'
        builder.save(str(gradient_graph_path))


if __name__ == '__main__':
    unittest.main()
