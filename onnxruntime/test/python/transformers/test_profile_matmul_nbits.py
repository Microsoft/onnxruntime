# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import onnx
from onnx import TensorProto
from profile_matmul_nbits import build_model


class TestProfileMatMulNBits(unittest.TestCase):
    def test_model_ir_matches_opset(self):
        model = onnx.load_from_string(build_model(256, 16, 32, 4, TensorProto.FLOAT16))
        self.assertEqual(model.ir_version, 8)
        onnx.checker.check_model(model)
