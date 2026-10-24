# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Tests that bfloat16 tensors flow through the standard Python bindings —
# `session.run`, `OrtValue.ortvalue_from_numpy`, and `OrtValue.numpy()` —
# using the `ml_dtypes.bfloat16` numpy extension dtype. NumPy has no native
# bfloat16, so the type is registered at pybind init via `ml_dtypes` and
# added to the two type maps in onnxruntime_pybind_mlvalue.cc.

from __future__ import annotations

import unittest

import ml_dtypes
import numpy as np
from onnx import TensorProto, helper

import onnxruntime as onnxrt

# The highest ai.onnx opset that has been *released* by the installed onnx
# package. onnx.defs.onnx_opset_version() returns the in-development next
# opset, which ORT's loader rejects. Kept in sync with the pattern used in
# onnxruntime_test_python_iobinding.py.
_LAST_RELEASED_AI_ONNX_OPSET = max(v for (d, v) in helper.OP_SET_ID_VERSION_MAP if d == "ai.onnx")


def _make_identity_bf16_model() -> bytes:
    x = helper.make_tensor_value_info("X", TensorProto.BFLOAT16, [None])
    y = helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [None])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph([node], "bf16_identity", [x], [y], [])
    model = helper.make_model(
        graph,
        producer_name="onnxruntime-test",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", _LAST_RELEASED_AI_ONNX_OPSET)],
    )
    return model.SerializeToString()


def _make_add_bf16_model() -> bytes:
    a = helper.make_tensor_value_info("A", TensorProto.BFLOAT16, [None])
    b = helper.make_tensor_value_info("B", TensorProto.BFLOAT16, [None])
    y = helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [None])
    node = helper.make_node("Add", ["A", "B"], ["Y"])
    graph = helper.make_graph([node], "bf16_add", [a, b], [y], [])
    model = helper.make_model(
        graph,
        producer_name="onnxruntime-test",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", _LAST_RELEASED_AI_ONNX_OPSET)],
    )
    return model.SerializeToString()


class TestBFloat16Numpy(unittest.TestCase):
    # ------- unit-level: OrtValue <-> numpy bfloat16 round-trip -------

    def test_ortvalue_from_numpy_bfloat16_roundtrip(self):
        data = np.array([1.0, -2.5, 0.0, 3.5, 42.0], dtype=ml_dtypes.bfloat16)
        ort_value = onnxrt.OrtValue.ortvalue_from_numpy(data)
        self.assertTrue(ort_value.is_tensor())
        self.assertEqual(ort_value.element_type(), TensorProto.BFLOAT16)
        self.assertEqual(list(ort_value.shape()), list(data.shape))
        recovered = ort_value.numpy()
        self.assertEqual(recovered.dtype, ml_dtypes.bfloat16)
        # Round-trip is bit-exact (no precision loss because the source
        # buffer was already bfloat16).
        np.testing.assert_array_equal(recovered, data)

    def test_ortvalue_shape_and_type_bfloat16(self):
        ort_value = onnxrt.OrtValue.ortvalue_from_shape_and_type([2, 3], ml_dtypes.bfloat16)
        self.assertEqual(ort_value.element_type(), TensorProto.BFLOAT16)
        self.assertEqual(list(ort_value.shape()), [2, 3])
        arr = ort_value.numpy()
        self.assertEqual(arr.dtype, ml_dtypes.bfloat16)
        self.assertEqual(arr.shape, (2, 3))

    # ------- integration: session.run with bfloat16 numpy inputs -------

    def test_session_run_bf16_identity(self):
        sess = onnxrt.InferenceSession(_make_identity_bf16_model(), providers=["CPUExecutionProvider"])
        # Identity uses type constraint V (all types) and does no math, so
        # it exercises the input/output plumbing without needing a BF16
        # arithmetic kernel on CPU.
        x = np.array([0.0, 1.5, -2.75, 8.0], dtype=ml_dtypes.bfloat16)
        outputs = sess.run(None, {"X": x})
        self.assertEqual(len(outputs), 1)
        y = outputs[0]
        self.assertEqual(y.dtype, ml_dtypes.bfloat16)
        np.testing.assert_array_equal(y, x)

    def test_session_run_bf16_add_on_cuda(self):
        if "CUDAExecutionProvider" not in onnxrt.get_available_providers():
            self.skipTest("CUDA execution provider is not available.")

        sess = onnxrt.InferenceSession(
            _make_add_bf16_model(),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=ml_dtypes.bfloat16)
        b = np.array([0.5, -1.0, 2.5, 0.25], dtype=ml_dtypes.bfloat16)
        outputs = sess.run(None, {"A": a, "B": b})
        y = outputs[0]
        self.assertEqual(y.dtype, ml_dtypes.bfloat16)
        # Expected result computed in float32 and cast back to bfloat16 to
        # match the kernel's accumulation semantics.
        expected = (a.astype(np.float32) + b.astype(np.float32)).astype(ml_dtypes.bfloat16)
        np.testing.assert_array_equal(y, expected)

    def test_session_run_bf16_output_dtype_preserved(self):
        # The output numpy array should carry ml_dtypes.bfloat16 (not be
        # silently viewed as uint16 or float16) so downstream code sees
        # the correct semantic dtype.
        sess = onnxrt.InferenceSession(_make_identity_bf16_model(), providers=["CPUExecutionProvider"])
        x = np.arange(8, dtype=np.float32).astype(ml_dtypes.bfloat16)
        (y,) = sess.run(None, {"X": x})
        self.assertIs(y.dtype.type, ml_dtypes.bfloat16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
